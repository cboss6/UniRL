"""Metric naming for rollout versus recomputed old-policy log-probabilities."""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict

from .contract import ExactLogprobResult


def exact_result_metrics(result: ExactLogprobResult) -> Dict[str, float]:
    raw = asdict(result)
    return {f"parity/{key}": float(value) for key, value in raw.items() if isinstance(value, (bool, int, float))}


def format_exact_result(result: ExactLogprobResult) -> str:
    return (
        f"tokens={result.token_count} finite={result.finite} "
        f"torch_equal={result.torch_equal} mismatch_count={result.mismatch_count} "
        f"max_absdiff_fp32={result.max_absdiff_fp32!r} "
        f"k3_mean={result.k3_mean!r} k3_max={result.k3_max!r}"
    )


__all__ = ["exact_result_metrics", "format_exact_result"]
