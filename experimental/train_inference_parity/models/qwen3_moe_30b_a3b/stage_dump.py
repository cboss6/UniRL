"""Actor-side tensor dumps for the Qwen3-MoE parity experiment."""

from __future__ import annotations

import os
from pathlib import Path

import torch


def install_actor_stage_dump(transformer) -> None:
    root = os.environ.get("UNIRL_PARITY_DUMP_DIR")
    if not root or getattr(transformer, "_unirl_parity_dump_installed", False):
        return
    rank = int(os.environ.get("RANK", "0"))
    side = "veomni" if type(transformer).__module__.startswith("veomni.") else "fsdp"
    directory = Path(root) / os.environ.get("UNIRL_PARITY_DUMP_RUN_ID", "run") / side / f"rank{rank:02d}"
    directory.mkdir(parents=True, exist_ok=True)
    calls: dict[tuple[int, str], int] = {}

    def emit(layer: int, stage: str, value) -> None:
        if isinstance(value, tuple):
            value = value[0]
        if not isinstance(value, torch.Tensor):
            return
        key = (layer, stage)
        call = calls.get(key, 0)
        calls[key] = call + 1
        torch.save(
            value.detach().contiguous().cpu(),
            directory / f"layer{layer:02d}.call{call:06d}.{stage}.pt",
        )

    for layer_index, layer in enumerate(transformer.model.layers):
        layer.register_forward_pre_hook(
            lambda _owner, inputs, index=layer_index: emit(
                index,
                "layer_input",
                inputs[0],
            )
        )
        layer.register_forward_hook(
            lambda _owner, _inputs, output, index=layer_index: emit(
                index,
                "layer_output",
                output,
            )
        )
    transformer._unirl_parity_dump_installed = True


__all__ = ["install_actor_stage_dump"]
