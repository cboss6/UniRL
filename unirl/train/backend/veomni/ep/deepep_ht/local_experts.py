"""Rank-local VeOmni grouped GEMM after DeepEP-HT dispatch."""

from __future__ import annotations

import torch


class LocalExpertsError(RuntimeError):
    """Raised when DeepEP recv metadata cannot be packed into VeOmni GEMM."""


def _as_local(parameter: torch.Tensor) -> torch.Tensor:
    to_local = getattr(parameter, "to_local", None)
    return to_local() if to_local is not None else parameter


def _prepare_grouped_rows(
    hidden_states: torch.Tensor,
    local_expert_ids: torch.Tensor,
    local_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens, topk = local_expert_ids.shape
    flat_ids = local_expert_ids.reshape(-1)
    valid = flat_ids >= 0
    valid_ids = flat_ids[valid]
    source_tokens = (
        torch.arange(tokens, device=hidden_states.device).unsqueeze(1).expand(tokens, topk).reshape(-1)
    )[valid]
    order = torch.argsort(valid_ids, stable=True)
    sorted_ids = valid_ids[order]
    dispatched = hidden_states[source_tokens[order]].contiguous()
    counts = torch.bincount(sorted_ids, minlength=local_experts).to(torch.int64)
    return dispatched, counts, source_tokens[order], order


def rank_local_expert_partial(
    hidden_states: torch.Tensor,
    local_expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
) -> torch.Tensor:
    """Run VeOmni merged-FC1 grouped GEMM on packed DeepEP recv tokens.

    ``local_expert_ids`` uses ``-1`` for slots that did not land on this rank.
    Routing weights are applied after FC2; the returned tensor is the BF16
    rank-local partial expected by DeepEP ``Buffer.combine``.
    """

    if hidden_states.dtype != torch.bfloat16:
        raise LocalExpertsError(f"local experts require BF16 hidden states, got {hidden_states.dtype}")
    gate_up_proj = _as_local(gate_up_proj)
    down_proj = _as_local(down_proj)
    if gate_up_proj.dtype != torch.bfloat16 or down_proj.dtype != torch.bfloat16:
        raise LocalExpertsError("rank-local expert weights must both be BF16")
    if local_expert_ids.dim() != 2:
        raise LocalExpertsError("DeepEP local expert IDs must be a two-dimensional tensor")
    local_expert_ids = local_expert_ids.to(torch.int64)
    if routing_weights.shape != local_expert_ids.shape:
        raise LocalExpertsError("DeepEP routing weights and local expert IDs must match")
    local_experts = int(gate_up_proj.shape[0])
    invalid = (local_expert_ids < -1) | (local_expert_ids >= local_experts)
    if bool(invalid.any()):
        raise LocalExpertsError("DeepEP returned a local expert ID outside [-1, local_experts)")

    tokens, hidden = hidden_states.shape[0], hidden_states.shape[-1]
    dispatched, counts, packed_sources, order = _prepare_grouped_rows(
        hidden_states, local_expert_ids, local_experts
    )
    if dispatched.shape[0] == 0:
        zero = (
            hidden_states.sum() * 0
            + routing_weights.sum() * 0
            + gate_up_proj.sum() * 0
            + down_proj.sum() * 0
        )
        return zero + torch.zeros((tokens, hidden), dtype=torch.bfloat16, device=hidden_states.device)

    from veomni.distributed.moe import EPMergedFc1GroupGemm

    cumsum = torch.cumsum(counts, dim=0).to(device=dispatched.device, dtype=torch.int64)
    expert_out = EPMergedFc1GroupGemm.apply(dispatched, cumsum, gate_up_proj, down_proj)
    flat_weights = routing_weights.reshape(-1)
    valid = local_expert_ids.reshape(-1) >= 0
    sorted_weights = flat_weights[valid][order].float()
    weighted = expert_out.float() * sorted_weights.unsqueeze(-1)
    index = packed_sources.unsqueeze(-1).expand_as(weighted)
    partial = torch.zeros((tokens, hidden), dtype=torch.float32, device=hidden_states.device)
    partial = partial.scatter_add(0, index, weighted)
    return partial.to(torch.bfloat16)


__all__ = ["LocalExpertsError", "rank_local_expert_partial"]
