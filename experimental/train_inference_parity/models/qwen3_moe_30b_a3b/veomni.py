"""VeOmni EP4 actor integration using public providers and core DeepEP transport."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from unirl.models.qwen3_moe.bundle import Qwen3MoeBundle
from unirl.models.types.meta_init import finalize_meta_init
from unirl.utils.dtypes import canonical_torch_dtype_name, parse_torch_dtype

from .contract import validate_model_config
from .fsdp import _RmsNorm
from .fsdp import install as install_actor_contract

MOE_IMPLEMENTATION = "parity_deepep_ht"
RMS_IMPLEMENTATION = "parity_vllm_bi"
ROTARY_IMPLEMENTATION = "parity_vllm_exact"
_REGISTERED = False
_DISPATCHER = None


def _providers():
    from unirl_train_inference_parity_vllm.common import providers

    return providers


def _grouped_linear(inputs, weights, counts, *, public):
    outputs = []
    offset = 0
    for expert in range(int(weights.shape[0])):
        count = int(counts[expert].item())
        if count:
            rows = inputs[offset : offset + count].contiguous()
            outputs.append(
                _providers().linear(rows, weights[expert], None) if public else F.linear(rows, weights[expert])
            )
        offset += count
    return torch.cat(outputs, dim=0)


def _local_slots(hidden, local_ids, routing, gate_up, down, *, public):
    tokens, topk = local_ids.shape
    hidden_size = int(hidden.shape[-1])
    flat_ids = local_ids.reshape(-1)
    valid = flat_ids >= 0
    positions = valid.nonzero().flatten()
    if not int(positions.numel()):
        return torch.zeros(
            (tokens, topk, hidden_size),
            dtype=torch.bfloat16,
            device=hidden.device,
        )
    ids = flat_ids[valid]
    source_tokens = torch.arange(tokens, device=hidden.device)[:, None].expand(tokens, topk).reshape(-1)[valid]
    order = torch.argsort(ids, stable=True)
    sorted_ids = ids[order]
    dispatched = hidden.index_select(0, source_tokens[order]).contiguous()
    counts = torch.bincount(sorted_ids, minlength=gate_up.shape[0])
    fc1 = _grouped_linear(dispatched, gate_up, counts, public=public)
    gate, up = fc1.chunk(2, dim=-1)
    activation = F.silu(gate) * up
    fc2 = _grouped_linear(activation, down, counts, public=public)
    weights = routing.reshape(-1)[valid][order].float()
    weighted = (fc2.float() * weights[:, None]).to(torch.bfloat16)
    slots = torch.zeros(
        (tokens * topk, hidden_size),
        dtype=torch.bfloat16,
        device=hidden.device,
    )
    slots[positions[order]] = weighted
    return slots.view(tokens, topk, hidden_size)


class _LocalExperts(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, local_ids, routing, gate_up, down):
        ctx.save_for_backward(hidden, local_ids, routing, gate_up, down)
        return _local_slots(
            hidden,
            local_ids,
            routing,
            gate_up,
            down,
            public=True,
        )

    @staticmethod
    def backward(ctx, grad_output):
        hidden, local_ids, routing, gate_up, down = ctx.saved_tensors
        needs = ctx.needs_input_grad
        sources = (hidden, routing, gate_up, down)
        rebuilt = [
            tensor.detach().requires_grad_(needed)
            for tensor, needed in zip(
                sources,
                (needs[0], needs[2], needs[3], needs[4]),
                strict=True,
            )
        ]
        with torch.enable_grad():
            output = _local_slots(
                rebuilt[0],
                local_ids,
                rebuilt[1],
                rebuilt[2],
                rebuilt[3],
                public=False,
            )
            requested = [tensor for tensor in rebuilt if tensor.requires_grad]
            computed = torch.autograd.grad(
                output,
                requested,
                grad_output,
                allow_unused=True,
            )
        iterator = iter(computed)
        gradients = [next(iterator) if tensor.requires_grad else None for tensor in rebuilt]
        return gradients[0], None, gradients[1], gradients[2], gradients[3]


def _rank_partial(slots, local_ids):
    result = torch.zeros(
        (slots.shape[0], slots.shape[-1]),
        dtype=torch.float32,
        device=slots.device,
    )
    zero = torch.zeros_like(result)
    for slot in range(int(slots.shape[1])):
        result = result + torch.where(
            (local_ids[:, slot] >= 0)[:, None],
            slots[:, slot].float(),
            zero,
        )
    return result.to(torch.bfloat16)


def _as_local(parameter):
    method = getattr(parameter, "to_local", None)
    return method() if method is not None else parameter


def _moe_factory():
    def adapter(self, hidden_states, top_k_index, top_k_weights):
        from veomni.distributed.parallel_state import get_parallel_state

        dispatcher = _dispatcher()
        invocation, received, local_ids, received_weights = dispatcher.dispatch(
            group=get_parallel_state().ep_group,
            num_experts=128,
            hidden_states=hidden_states,
            topk_indices=top_k_index,
            routing_weights=top_k_weights.float(),
            gate_up_proj=_as_local(self.gate_up_proj),
            down_proj=_as_local(self.down_proj),
        )
        slots = _LocalExperts.apply(
            received,
            local_ids.long(),
            received_weights.float(),
            _as_local(self.gate_up_proj),
            _as_local(self.down_proj),
        )
        return dispatcher.combine(invocation, _rank_partial(slots, local_ids))

    return adapter


def _rms_factory():
    def rms_norm(hidden, weight, eps):
        return _RmsNorm.apply(hidden, weight, float(eps))

    return rms_norm


def _rotary_factory():
    def apply(q, k, cosine, sine, position_ids=None, unsqueeze_dim=1):
        del position_ids
        if unsqueeze_dim != 1:
            raise ValueError("Qwen3 parity RoPE requires unsqueeze_dim=1")

        def rotate(value):
            cos = cosine.unsqueeze(1).to(value.dtype)
            sin = sine.unsqueeze(1).to(value.dtype)
            first, second = value.chunk(2, dim=-1)
            half = value.shape[-1] // 2
            return torch.cat(
                (
                    first * cos[..., :half] - second * sin[..., :half],
                    second * cos[..., :half] + first * sin[..., :half],
                ),
                dim=-1,
            )

        return rotate(q), rotate(k)

    return apply


def _dispatcher():
    global _DISPATCHER
    if _DISPATCHER is None:
        from unirl.train.backend.veomni.ep.deepep_ht.config import DeepEPHTConfig
        from unirl.train.backend.veomni.ep.deepep_ht.dispatcher import (
            DeepEPDispatcher,
        )

        _DISPATCHER = DeepEPDispatcher(DeepEPHTConfig.from_env())
    return _DISPATCHER


def _install_generated_model_patches() -> None:
    try:
        from veomni.models.transformers.qwen3_moe.generated import (
            patched_modeling_qwen3_moe_gpu as modeling,
        )
    except ModuleNotFoundError:
        return

    def router_forward(self, hidden_states):
        hidden = hidden_states.reshape(-1, self.hidden_dim)
        logits = F.linear(hidden, self.weight)
        probabilities = F.softmax(logits.float(), dim=-1)
        weights, ids = torch.topk(probabilities, self.top_k, dim=-1)
        if self.norm_topk_prob:
            weights = weights / weights.sum(dim=-1, keepdim=True)
        return logits, weights.float(), ids

    def frequency_forward(self, x, position_ids):
        base = self.config.rope_parameters["rope_theta"]
        dimension = getattr(self.config, "head_dim", None) or (
            self.config.hidden_size // self.config.num_attention_heads
        )
        inverse = 1.0 / (base ** (torch.arange(0, dimension, 2, dtype=torch.float32, device=x.device) / dimension))
        frequencies = (inverse[None, :, None] * position_ids[:, None, :].float()).transpose(1, 2)
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        return (
            (embedding.cos() * self.attention_scaling).to(x.dtype),
            (embedding.sin() * self.attention_scaling).to(x.dtype),
        )

    modeling.Qwen3MoeTopKRouter.forward = router_forward
    modeling.Qwen3MoeRotaryEmbedding.forward = frequency_forward

    def block_forward(self, hidden_states):
        batch, sequence, hidden_size = hidden_states.shape
        hidden = hidden_states.reshape(-1, hidden_size)
        _logits, routing_weights, selected_experts = self.gate(hidden)
        kernel = modeling.veomni_moe_experts_forward.bound_kernel()
        if kernel is None:
            raise RuntimeError("parity VeOmni MoE OpSlot is not bound")
        output = kernel(
            self.experts,
            hidden,
            selected_experts,
            routing_weights.float(),
        )
        return output.view(batch, sequence, hidden_size)

    modeling.Qwen3MoeSparseMoeBlock.forward = block_forward


def _install_veomni_compatibility() -> None:
    from veomni.ops.kernels import moe as veomni_moe

    current = veomni_moe.apply_veomni_fused_moe_patch
    if not getattr(current, "_unirl_parity_bridge", False):

        def apply_veomni_fused_moe_patch(fused_moe_kernel: str = "triton"):
            if fused_moe_kernel == MOE_IMPLEMENTATION:
                return None
            return current(fused_moe_kernel=fused_moe_kernel)

        apply_veomni_fused_moe_patch._unirl_parity_bridge = True
        veomni_moe.apply_veomni_fused_moe_patch = apply_veomni_fused_moe_patch

    import inspect

    from transformers import masking_utils

    create_mask = masking_utils.create_causal_mask
    if "cache_position" not in inspect.signature(create_mask).parameters:

        def create_causal_mask(*args, cache_position=None, **kwargs):
            del cache_position
            return create_mask(*args, **kwargs)

        masking_utils.create_causal_mask = create_causal_mask


def register() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    install_actor_contract()
    _install_veomni_compatibility()
    from veomni.ops.kernel_registry import (
        KERNEL_REGISTRY,
        HardwareRequirement,
        KernelSpec,
    )

    entries = (
        (MOE_IMPLEMENTATION, "moe_experts", "standard", _moe_factory),
        (RMS_IMPLEMENTATION, "rms_norm", "standard", _rms_factory),
        (ROTARY_IMPLEMENTATION, "rotary_pos_emb", "full", _rotary_factory),
    )
    for name, op_name, variant, factory in entries:
        available = set(KERNEL_REGISTRY.list_available(op_name, variant))
        if name not in available:
            KERNEL_REGISTRY.register(
                KernelSpec(
                    name=name,
                    op_name=op_name,
                    variant=variant,
                    factory=factory,
                    hardware=HardwareRequirement(
                        device_type="gpu",
                        min_compute_capability=90,
                    ),
                    description=f"UniRL parity public reference: {name}",
                )
            )
    _install_generated_model_patches()
    _REGISTERED = True


class ParityQwen3MoeBundle(Qwen3MoeBundle):
    @classmethod
    def from_config(cls, config):
        register()
        from veomni.arguments import OpsImplementationConfig
        from veomni.models.auto import build_foundation_model

        from unirl.train.backend.veomni import _compat

        _compat.ensure_qwen3_moe_installed()
        dtype = parse_torch_dtype(
            str(config.model_precision),
            field_name="ParityQwen3MoeBundle.model_precision",
        )
        transformer = build_foundation_model(
            config_path=config.pretrained_model_ckpt_path,
            weights_path=None,
            torch_dtype=canonical_torch_dtype_name(dtype, field_name="model_precision"),
            init_device="meta",
            ops_implementation=OpsImplementationConfig(
                attn_implementation=config.attn_implementation,
                moe_implementation=MOE_IMPLEMENTATION,
                cross_entropy_loss_implementation="eager",
                rms_norm_implementation=RMS_IMPLEMENTATION,
                swiglu_mlp_implementation="eager",
                rotary_pos_emb_implementation=ROTARY_IMPLEMENTATION,
                load_balancing_loss_implementation="eager",
            ),
        )
        transformer = finalize_meta_init(transformer, dtype=dtype)
        validate_model_config(transformer.config)
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            config.pretrained_model_ckpt_path,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        bundle = cls(
            transformer=transformer,
            tokenizer=tokenizer,
            dtype=dtype,
            device=torch.device("cuda"),
            pretrained_path=config.pretrained_model_ckpt_path,
            ep_comm_backend="parity",
        )
        bundle._transformer_weights_path = config.pretrained_model_ckpt_path
        return bundle

    def close_expert_parallel(self) -> None:
        from unirl.train.backend.veomni.ep.deepep_ht.dispatcher import (
            close_deepep_ht_buffers,
        )

        close_deepep_ht_buffers()


__all__ = ["ParityQwen3MoeBundle", "register"]
