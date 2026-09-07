"""Qwen3MoeBundle — VeOmni-patched Qwen3-MoE causal LM + tokenizer."""

from __future__ import annotations

import importlib
import logging
import os
import sys
from typing import Any, Optional

import torch
import torch.nn as nn

from unirl.models.types.bundle import Bundle
from unirl.models.types.meta_init import finalize_meta_init
from unirl.utils.dtypes import canonical_torch_dtype_name, parse_torch_dtype

logger = logging.getLogger(__name__)

_EP_COMM_BACKENDS = frozenset({"torch", "deepep_ht", "unimatch"})
_UNIMATCH_MOE_IMPLEMENTATION = "uexact_deepep_ht"
_UNIMATCH_RMS_IMPLEMENTATION = "unimatch_bi"
_UNIMATCH_ROTARY_IMPLEMENTATION = "unimatch_vllm_exact"


def _resolve_ops_implementations(
    ep_comm_backend: str,
    moe_implementation: Optional[str],
) -> tuple[str, str, str, str]:
    backend = str(ep_comm_backend).strip().lower()
    if backend not in _EP_COMM_BACKENDS:
        raise ValueError(
            "Qwen3MoeBundle.ep_comm_backend must be 'torch', 'deepep_ht', or "
            f"'unimatch'; got {ep_comm_backend!r}"
        )

    requested_moe = None if moe_implementation is None else str(moe_implementation).strip()
    if requested_moe == "":
        raise ValueError("Qwen3MoeBundle.moe_implementation cannot be empty")
    if backend == "unimatch":
        if requested_moe not in {None, _UNIMATCH_MOE_IMPLEMENTATION}:
            raise ValueError(
                "ep_comm_backend=unimatch requires VeOmni's exact registered MoE "
                f"OpSlot {_UNIMATCH_MOE_IMPLEMENTATION!r}; got {requested_moe!r}"
            )
        return (
            backend,
            _UNIMATCH_MOE_IMPLEMENTATION,
            _UNIMATCH_RMS_IMPLEMENTATION,
            _UNIMATCH_ROTARY_IMPLEMENTATION,
        )

    resolved_moe = requested_moe or "fused_triton"
    if backend == "deepep_ht" and resolved_moe not in {"fused_triton", "fused"}:
        raise ValueError(
            "ep_comm_backend=deepep_ht keeps VeOmni fused_triton compute; "
            f"got moe_implementation={resolved_moe!r}"
        )
    return backend, resolved_moe, "eager", "eager"


def _register_unimatch_veomni() -> None:
    """Import and explicitly register the adaptor before VeOmni binds OpSlots."""

    try:
        bootstrap = importlib.import_module("unimatch.adaptor.veomni.bootstrap")
    except Exception as exc:
        raise RuntimeError(
            "ep_comm_backend=unimatch requires an installed UniMatch VeOmni "
            "adaptor at unimatch.adaptor.veomni.bootstrap"
        ) from exc

    expected = {
        "KERNEL_NAME": _UNIMATCH_MOE_IMPLEMENTATION,
        "RMS_KERNEL_NAME": _UNIMATCH_RMS_IMPLEMENTATION,
        "ROTARY_KERNEL_NAME": _UNIMATCH_ROTARY_IMPLEMENTATION,
    }
    actual = {name: getattr(bootstrap, name, None) for name in expected}
    if actual != expected:
        raise RuntimeError(
            "UniMatch VeOmni adaptor exports incompatible OpSlot names: "
            f"expected={expected}, actual={actual}"
        )
    register = getattr(bootstrap, "register", None)
    if not callable(register):
        raise RuntimeError("UniMatch VeOmni bootstrap does not expose callable register()")
    register()
    if not os.environ.get("UNIMATCH_STAGE_DUMP_DIR"):
        return

    install_stage_dump_hook = getattr(bootstrap, "install_stage_dump_hook", None)
    if not callable(install_stage_dump_hook):
        raise RuntimeError("UniMatch VeOmni bootstrap does not expose install_stage_dump_hook()")
    try:
        from unimatch.diagnostics.stage_dump import emit_stage
    except Exception as exc:
        raise RuntimeError(
            "UniMatch VeOmni stage diagnostics require "
            "unimatch.diagnostics.stage_dump.emit_stage"
        ) from exc

    def emit_veomni_stage(
        stage,
        tensor,
        *,
        layer,
        call,
        record_id=None,
        metadata=None,
    ):
        del record_id
        return emit_stage(
            "veomni",
            stage,
            tensor,
            layer=layer,
            call=call,
            metadata=metadata,
        )

    install_stage_dump_hook(emit_veomni_stage)


class Qwen3MoeBundle(Bundle):
    """VeOmni Qwen3-MoE transformer + tokenizer (meta-init, EP-capable)."""

    def __init__(
        self,
        *,
        transformer: nn.Module,
        tokenizer: Any,
        dtype: torch.dtype,
        device: torch.device,
        pretrained_path: str,
        ep_comm_backend: str = "torch",
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.tokenizer = tokenizer
        self.dtype = dtype
        self.device = device
        self.pretrained_path = pretrained_path
        self.ep_comm_backend = ep_comm_backend
        self._ep_buffers_closed = False

    def prepare_for_expert_parallel(self) -> None:
        """Backend hook when ``ep_size > 1``."""
        if not callable(getattr(self.transformer, "get_parallel_plan", None)):
            raise ValueError(
                "Qwen3MoeBundle.prepare_for_expert_parallel: transformer lacks "
                "get_parallel_plan(); rebuild with an EP-capable MoE implementation"
            )

    def close_expert_parallel(self) -> None:
        """Release process-local DeepEP buffers before distributed teardown."""

        if self._ep_buffers_closed:
            return
        self._ep_buffers_closed = True
        if self.ep_comm_backend == "deepep_ht":
            from unirl.train.backend.veomni.ep.deepep_ht import close_deepep_ht_buffers

            close_deepep_ht_buffers()
            return
        if self.ep_comm_backend != "unimatch":
            return

        bootstrap = sys.modules.get("unimatch.adaptor.veomni.bootstrap")
        shutdown = getattr(bootstrap, "shutdown", None)
        if callable(shutdown):
            shutdown()
            return
        # Avoid importing accelerator code during teardown; use only the
        # already-loaded dispatcher as a compatibility fallback.
        dispatcher = sys.modules.get("unimatch.adaptor.veomni.dispatcher")
        if dispatcher is None:
            return
        close = getattr(dispatcher, "close_deepep_ht_buffers", None)
        if callable(close):
            close()
            return
        cache = getattr(dispatcher, "_BUFFER_CACHE", None)
        clear = getattr(cache, "clear", None)
        if callable(clear):
            clear()
        else:
            logger.warning("UniMatch VeOmni dispatcher exposes no DeepEP buffer cleanup hook")

    @classmethod
    def from_config(
        cls,
        config: Any = None,
        *,
        pretrained_model_ckpt_path: Optional[str] = None,
        tokenizer_ckpt_path: Optional[str] = None,
        model_precision: str = "bf16",
        moe_implementation: Optional[str] = None,
        ep_comm_backend: str = "torch",
        attn_implementation: str = "flash_attention_2",
        meta_init_transformer: bool = True,
        trust_remote_code: bool = True,
        tokenizer: Any = None,
    ) -> "Qwen3MoeBundle":
        """Build the VeOmni Qwen3-MoE transformer (on meta) + tokenizer."""
        if config is not None:
            pretrained_model_ckpt_path = config.pretrained_model_ckpt_path
            tokenizer_ckpt_path = getattr(config, "tokenizer_ckpt_path", None)
            model_precision = getattr(config, "model_precision", "bf16")
            attn_implementation = getattr(config, "attn_implementation", None) or attn_implementation
            moe_implementation = getattr(config, "moe_implementation", None) or moe_implementation
            ep_comm_backend = getattr(config, "ep_comm_backend", None) or ep_comm_backend
            meta_init_transformer = bool(getattr(config, "meta_init_transformer", True))
            trust_remote_code = bool(getattr(config, "trust_remote_code", True))
        if pretrained_model_ckpt_path is None:
            raise ValueError("Qwen3MoeBundle.from_config: pretrained_model_ckpt_path is required")
        if not meta_init_transformer:
            raise ValueError(
                "Qwen3MoeBundle requires meta_init_transformer=true: VeOmniBackend "
                "materializes and loads this EP model only after parallelization."
            )
        (
            ep_comm_backend,
            moe_implementation,
            rms_norm_implementation,
            rotary_pos_emb_implementation,
        ) = _resolve_ops_implementations(ep_comm_backend, moe_implementation)

        from unirl.train.backend.veomni import _compat

        _compat.ensure_qwen3_moe_installed()
        if ep_comm_backend == "deepep_ht":
            from unirl.train.backend.veomni.ep.deepep_ht import install_deepep_ht_patch

            install_deepep_ht_patch()
        elif ep_comm_backend == "unimatch":
            _register_unimatch_veomni()
        from veomni.arguments import OpsImplementationConfig
        from veomni.models.auto import build_foundation_model

        dtype = parse_torch_dtype(str(model_precision), field_name="Qwen3MoeBundle.model_precision")
        dtype_name = canonical_torch_dtype_name(dtype, field_name="Qwen3MoeBundle.model_precision")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ops = OpsImplementationConfig(
            attn_implementation=attn_implementation,
            moe_implementation=moe_implementation,
            cross_entropy_loss_implementation="eager",
            rms_norm_implementation=rms_norm_implementation,
            swiglu_mlp_implementation="eager",
            rotary_pos_emb_implementation=rotary_pos_emb_implementation,
            load_balancing_loss_implementation="eager",
        )
        transformer = build_foundation_model(
            config_path=pretrained_model_ckpt_path,
            weights_path=None,
            torch_dtype=dtype_name,
            init_device="meta",
            ops_implementation=ops,
        )
        transformer = finalize_meta_init(transformer, dtype=dtype)

        if tokenizer is None:
            from transformers import AutoTokenizer

            tok_path = tokenizer_ckpt_path or pretrained_model_ckpt_path
            tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=trust_remote_code)
            if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token

        bundle = cls(
            transformer=transformer,
            tokenizer=tokenizer,
            dtype=dtype,
            device=device,
            pretrained_path=pretrained_model_ckpt_path,
            ep_comm_backend=ep_comm_backend,
        )
        bundle._transformer_weights_path = pretrained_model_ckpt_path
        return bundle


__all__ = ["Qwen3MoeBundle"]
