"""Qwen3-MoE weight-layout validation used before online reload."""

from __future__ import annotations

_EXPERT_SUFFIXES = (
    ".gate_proj.weight",
    ".up_proj.weight",
    ".down_proj.weight",
)


def validate_exported_weight_names(names) -> None:
    names = tuple(str(name) for name in names)
    if not names:
        raise ValueError("parity weight export produced no tensors")
    expert_names = [name for name in names if ".experts." in name]
    if not expert_names:
        raise ValueError("parity Qwen3-MoE export contains no expert tensors")
    bad = [name for name in expert_names if not name.endswith(_EXPERT_SUFFIXES)]
    if bad:
        raise ValueError(f"unexpected Qwen3-MoE expert weight names: {bad[:8]}")


__all__ = ["validate_exported_weight_names"]
