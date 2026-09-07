"""VeOmni-specific sharded-state helpers."""

from __future__ import annotations

import logging
import os

import torch
from torch import Tensor, nn
from torch.distributed._tensor import DTensor

from unirl.train.backend.sharded_state import _maybe_dtensor_to_tensor

logger = logging.getLogger(__name__)


def clip_grad_norm(model: nn.Module, max_norm: float) -> Tensor:
    """Gradient clipping via VeOmni's FSDP2 clip (EP-aware under Phase 2)."""
    if os.environ.get(
        "UNIRL_VEOMNI_STREAMING_GRAD_NORM",
        "0",
    ) == "1":
        return _streaming_ep_grad_norm(model, max_norm)
    from unirl.train.backend.veomni import _compat

    _compat.ensure_installed()
    from veomni.distributed.fsdp2 import clip_grad_norm as _veomni_clip_grad_norm

    result = _veomni_clip_grad_norm(model, max_norm)
    return _maybe_dtensor_to_tensor(result)


@torch.no_grad()
def _streaming_local_sq_sum(parameters) -> Tensor:
    result = torch.zeros((), dtype=torch.float32, device="cuda")
    chunk_elements = 1 << 20
    for parameter in parameters:
        grad = parameter.grad
        if grad is None:
            continue
        local = grad.to_local() if isinstance(grad, DTensor) else grad
        flat = local.detach().reshape(-1)
        for begin in range(0, flat.numel(), chunk_elements):
            chunk = flat[begin : begin + chunk_elements].float()
            result.add_(torch.sum(chunk * chunk))
    return result


@torch.no_grad()
def _streaming_ep_grad_norm(model: nn.Module, max_norm: float) -> Tensor:
    import torch.distributed as dist

    from unirl.train.backend.veomni import _compat

    _compat.ensure_installed()
    from veomni.distributed.parallel_state import get_parallel_state

    ps = get_parallel_state()
    groups = model._extra_parallel_param_groups
    non_extra = _streaming_local_sq_sum(
        groups.get("non_extra_parallel", ())
    )
    if ps.fsdp_group is not None:
        dist.all_reduce(non_extra, op=dist.ReduceOp.SUM, group=ps.fsdp_group)

    total_sq = non_extra
    for name in ps.extra_parallel_names:
        value = _streaming_local_sq_sum(groups.get(name, ()))
        mesh = ps.extra_parallel_fsdp_device_mesh.get(name)
        if mesh is not None:
            fsdp_group = mesh[f"{name}_fsdp"].get_group()
            if fsdp_group is not None:
                dist.all_reduce(
                    value,
                    op=dist.ReduceOp.SUM,
                    group=fsdp_group,
                )
        if ps.extra_parallel_enabled(name):
            group = ps.extra_parallel_group(name)
            if group is not None:
                dist.all_reduce(value, op=dist.ReduceOp.SUM, group=group)
        total_sq = total_sq + value

    total_norm = torch.sqrt(total_sq)
    coefficient = torch.clamp(
        torch.tensor(
            float(max_norm),
            dtype=torch.float32,
            device=total_norm.device,
        )
        / (total_norm + 1e-6),
        max=1.0,
    )
    for parameter in model.parameters():
        grad = parameter.grad
        if grad is None:
            continue
        local = grad.to_local() if isinstance(grad, DTensor) else grad
        local.mul_(coefficient.to(dtype=local.dtype))
    return total_norm


def veomni_offload(model: nn.Module) -> None:
    """Move the parallelized model to CPU via VeOmni (reshards the root first)."""
    meta_names = [n for n, p in model.named_parameters() if p.is_meta]
    if meta_names:
        raise RuntimeError(
            f"veomni_offload: {len(meta_names)} params still on meta "
            f"(e.g. {meta_names[:4]}); VeOmniBackend v1 requires a fully-"
            "materialized trainable module."
        )
    from unirl.train.backend.veomni import _compat

    _compat.ensure_installed()
    from veomni.distributed.offloading import offload_model_to_cpu

    offload_model_to_cpu(model)
    logger.debug("veomni_offload: offloaded params/grads to CPU")


def veomni_onload(model: nn.Module, device: torch.device) -> None:
    """Move the parallelized model back to ``device`` via VeOmni."""
    from unirl.train.backend.veomni import _compat

    _compat.ensure_installed()
    from veomni.distributed.offloading import load_model_to_gpu

    load_model_to_gpu(model, device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    logger.debug("veomni_onload: onloaded params/grads to %s", device)


__all__ = [
    "clip_grad_norm",
    "veomni_offload",
    "veomni_onload",
]
