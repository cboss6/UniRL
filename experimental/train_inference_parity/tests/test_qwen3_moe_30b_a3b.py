from __future__ import annotations

from types import SimpleNamespace

import pytest

from experimental.train_inference_parity.models.qwen3_moe_30b_a3b.contract import (
    validate_model_config,
)


def _config(**overrides):
    values = {
        "hidden_size": 2048,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 768,
        "vocab_size": 151936,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_qwen3_contract_accepts_signed_shape() -> None:
    validate_model_config(_config())


def test_qwen3_contract_rejects_shape_drift() -> None:
    with pytest.raises(ValueError, match="hidden_size"):
        validate_model_config(_config(hidden_size=4096))
