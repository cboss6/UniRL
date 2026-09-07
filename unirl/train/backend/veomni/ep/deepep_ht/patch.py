"""Monkey-patch VeOmni fused-MoE EP transport onto DeepEP-HT."""

from __future__ import annotations

import logging
from typing import Callable, Optional

import torch

from unirl.train.backend.veomni.ep.deepep_ht.config import DeepEPHTConfig
from unirl.train.backend.veomni.ep.deepep_ht.dispatcher import DeepEPDispatcher
from unirl.train.backend.veomni.ep.deepep_ht.local_experts import (
    _as_local,
    rank_local_expert_partial,
)

logger = logging.getLogger(__name__)

_ORIGINAL: Optional[Callable] = None
_DISPATCHER: Optional[DeepEPDispatcher] = None
_INSTALLED = False

FusedMoEForward = Callable[..., torch.Tensor]


def _require_deep_ep() -> None:
    try:
        import deep_ep  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "ep_comm_backend=deepep_ht requires an importable `deep_ep` package "
            "with high-throughput Buffer.dispatch/combine"
        ) from exc


def _reshape_tokens(
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...]]:
    orig_shape = tuple(hidden_states.shape)
    hidden = hidden_states.reshape(-1, orig_shape[-1])
    tokens = hidden.shape[0]
    if selected_experts.numel() % tokens != 0:
        raise ValueError(
            "selected_experts numel "
            f"{selected_experts.numel()} is not divisible by token count {tokens}"
        )
    topk = selected_experts.numel() // tokens
    topk_idx = selected_experts.reshape(tokens, topk).to(torch.int64)
    topk_w = routing_weights.reshape(tokens, topk)
    return hidden, topk_idx, topk_w, orig_shape


def deepep_ht_fused_moe_forward(
    num_experts: int,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    hidden_states: torch.Tensor,
    fc1_1_weight: torch.Tensor | None,
    fc1_2_weight: torch.Tensor | None,
    fc2_weight: torch.Tensor,
    fc1_1_2_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """DeepEP-HT dispatch/combine around VeOmni fused-triton local experts."""

    if _DISPATCHER is None:
        raise RuntimeError("DeepEP-HT patch is not installed")
    if fc1_1_2_weight is None:
        if fc1_1_weight is None or fc1_2_weight is None:
            raise ValueError("DeepEP-HT requires merged fc1_1_2_weight or split fc1 weights")
        fc1_1_2_weight = torch.cat([fc1_1_weight, fc1_2_weight], dim=1)

    from veomni.distributed.parallel_state import get_parallel_state

    group = get_parallel_state().ep_group
    if group is None:
        raise RuntimeError("DeepEP-HT requires VeOmni ep_group; set backend.fsdp_cfg.ep_size>1")

    hidden, topk_idx, topk_w, orig_shape = _reshape_tokens(
        hidden_states, selected_experts, routing_weights
    )
    gate_up = _as_local(fc1_1_2_weight)
    down = _as_local(fc2_weight)
    invocation, recv_x, recv_ids, recv_weights = _DISPATCHER.dispatch(
        group=group,
        num_experts=int(num_experts),
        hidden_states=hidden,
        topk_indices=topk_idx,
        routing_weights=topk_w,
        gate_up_proj=gate_up,
        down_proj=down,
    )
    partial = rank_local_expert_partial(recv_x, recv_ids, recv_weights, gate_up, down)
    output = _DISPATCHER.combine(invocation, partial)
    return output.reshape(orig_shape)


def _wrapped_forward(
    num_experts: int,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    hidden_states: torch.Tensor,
    fc1_1_weight: torch.Tensor | None,
    fc1_2_weight: torch.Tensor | None,
    fc2_weight: torch.Tensor,
    fc1_1_2_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    if _ORIGINAL is None:
        raise RuntimeError("DeepEP-HT wrap lost the original fused MoE forward")
    from veomni.distributed.parallel_state import get_parallel_state

    if get_parallel_state().ep_enabled:
        return deepep_ht_fused_moe_forward(
            num_experts,
            routing_weights,
            selected_experts,
            hidden_states,
            fc1_1_weight,
            fc1_2_weight,
            fc2_weight,
            fc1_1_2_weight,
        )
    return _ORIGINAL(
        num_experts,
        routing_weights,
        selected_experts,
        hidden_states,
        fc1_1_weight,
        fc1_2_weight,
        fc2_weight,
        fc1_1_2_weight,
    )


def install_deepep_ht_patch(config: DeepEPHTConfig | None = None) -> None:
    """Wrap VeOmni ``group_gemm_fused_moe_forward`` before model bind.

    Must run after ``ensure_qwen3_moe_installed()`` (so ``veomni.ops`` exists)
    and before ``build_foundation_model``.
    """

    global _ORIGINAL, _DISPATCHER, _INSTALLED
    if _INSTALLED:
        return
    _require_deep_ep()
    resolved = DeepEPHTConfig.from_env() if config is None else config
    resolved.validate()

    import veomni.ops.kernels.moe as moe
    import veomni.ops.kernels.moe.group_gemm as group_gemm

    _ORIGINAL = group_gemm.group_gemm_fused_moe_forward
    _DISPATCHER = DeepEPDispatcher(resolved)
    group_gemm.group_gemm_fused_moe_forward = _wrapped_forward
    if getattr(moe, "_fused_moe_forward", None) is _ORIGINAL:
        moe._fused_moe_forward = _wrapped_forward
    _INSTALLED = True
    message = (
        "installed UniRL VeOmni DeepEP-HT wrap on group_gemm_fused_moe_forward "
        f"({resolved.manifest()})"
    )
    print(message, flush=True)
    logger.warning(message)


def is_deepep_ht_patch_installed() -> bool:
    return _INSTALLED


__all__ = [
    "deepep_ht_fused_moe_forward",
    "install_deepep_ht_patch",
    "is_deepep_ht_patch_installed",
]
