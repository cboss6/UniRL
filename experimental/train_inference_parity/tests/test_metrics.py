from __future__ import annotations

import torch

from experimental.train_inference_parity.contract import compare_logprobs_exact
from experimental.train_inference_parity.metrics import (
    exact_result_metrics,
    format_exact_result,
)


def test_metric_names_are_parity_scoped() -> None:
    values = torch.tensor([-0.5], dtype=torch.float32)
    result = compare_logprobs_exact(values, values)
    metrics = exact_result_metrics(result)
    assert metrics["parity/max_absdiff_fp32"] == 0.0
    assert metrics["parity/k3_max"] == 0.0
    assert "torch_equal=True" in format_exact_result(result)
