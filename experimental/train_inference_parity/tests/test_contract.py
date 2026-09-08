from __future__ import annotations

import pytest
import torch

from experimental.train_inference_parity.contract import (
    compare_logprobs_exact,
    require_exact,
)


def test_exact_contract_passes_equal_fp32_values() -> None:
    values = torch.tensor([-1.0, -2.0], dtype=torch.float32)
    result = compare_logprobs_exact(values, values.clone())
    assert result.passed
    assert result.max_absdiff_fp32 == 0.0
    assert result.k3_mean == 0.0
    assert result.k3_max == 0.0


def test_exact_contract_rejects_k3_false_zero() -> None:
    rollout = torch.tensor([-1.0], dtype=torch.float32)
    old = torch.nextafter(rollout, torch.zeros_like(rollout))
    result = compare_logprobs_exact(old, rollout)
    assert result.k3_mean == 0.0
    assert result.max_absdiff_fp32 > 0.0
    assert not result.passed
    with pytest.raises(RuntimeError, match="mismatch_count=1"):
        require_exact(result, context="test")
