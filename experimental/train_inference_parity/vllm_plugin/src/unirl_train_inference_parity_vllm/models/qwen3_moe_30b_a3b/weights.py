"""Weight-cache lifecycle helpers for online vLLM reloads."""

from __future__ import annotations


def invalidate_transformed_weights(model) -> None:
    for module in model.modules():
        experts = getattr(module, "experts", None)
        if experts is not None and hasattr(experts, "_unirl_parity_w2_column"):
            experts._unirl_parity_w2_column = None


__all__ = ["invalidate_transformed_weights"]
