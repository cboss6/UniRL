"""Fail-closed contracts shared by parity algorithms and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ExactLogprobResult:
    token_count: int
    mismatch_count: int
    max_absdiff_fp32: float
    k3_mean: float
    k3_max: float
    finite: bool
    torch_equal: bool

    @property
    def passed(self) -> bool:
        return (
            self.finite
            and self.torch_equal
            and self.mismatch_count == 0
            and self.max_absdiff_fp32 == 0.0
            and self.k3_mean == 0.0
            and self.k3_max == 0.0
        )


def compare_logprobs_exact(
    old_logprobs: torch.Tensor,
    rollout_logprobs: torch.Tensor,
) -> ExactLogprobResult:
    """Compare identically shaped log-probabilities after explicit FP32 conversion."""
    if old_logprobs.shape != rollout_logprobs.shape:
        raise ValueError(
            "parity comparison requires identical shapes: "
            f"old={tuple(old_logprobs.shape)} rollout={tuple(rollout_logprobs.shape)}"
        )
    old = old_logprobs.detach().float()
    rollout = rollout_logprobs.detach().to(device=old.device).float()
    if old.numel() == 0:
        raise ValueError("parity comparison requires at least one token")
    finite = bool(torch.isfinite(old).all().item() and torch.isfinite(rollout).all().item())
    mismatch_count = int((old != rollout).sum().item())
    absdiff = (old - rollout).abs()
    log_ratio = (old - rollout).clamp(min=-20.0, max=20.0)
    k3 = torch.expm1(log_ratio) - log_ratio
    return ExactLogprobResult(
        token_count=int(old.numel()),
        mismatch_count=mismatch_count,
        max_absdiff_fp32=float(absdiff.max().item()),
        k3_mean=float(k3.mean().item()),
        k3_max=float(k3.max().item()),
        finite=finite,
        torch_equal=torch.equal(old, rollout),
    )


def require_exact(result: ExactLogprobResult, *, context: str) -> None:
    if result.passed:
        return
    raise RuntimeError(
        f"{context} failed: finite={result.finite} "
        f"torch_equal={result.torch_equal} mismatch_count={result.mismatch_count} "
        f"max_absdiff_fp32={result.max_absdiff_fp32!r} "
        f"k3_mean={result.k3_mean!r} k3_max={result.k3_max!r}"
    )


__all__ = ["ExactLogprobResult", "compare_logprobs_exact", "require_exact"]
