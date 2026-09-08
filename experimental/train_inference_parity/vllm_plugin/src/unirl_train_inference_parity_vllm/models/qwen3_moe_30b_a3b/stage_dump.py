"""Minimal vLLM stage dumps for first-divergence diagnosis."""

from __future__ import annotations

import os
import re
import types
from pathlib import Path

import torch


def _emit(stage: str, value, *, layer: int, call: int) -> None:
    root = os.environ.get("UNIRL_PARITY_DUMP_DIR")
    if not root or not isinstance(value, torch.Tensor):
        return
    rank = int(os.environ.get("RANK", "0"))
    try:
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        rank = int(get_tensor_model_parallel_rank())
        if int(get_tensor_model_parallel_world_size()) == 1:
            rank = int(os.environ.get("UNIRL_ROLLOUT_DP_RANK", rank))
    except Exception:
        pass
    directory = Path(root) / os.environ.get("UNIRL_PARITY_DUMP_RUN_ID", "run") / "vllm" / f"rank{rank:02d}"
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(
        value.detach().contiguous().cpu(),
        directory / f"layer{layer:02d}.call{call:06d}.{stage}.pt",
    )


def install_stage_dump(module) -> None:
    if not os.environ.get("UNIRL_PARITY_DUMP_DIR"):
        return
    decoder = module.Qwen3MoeDecoderLayer
    original_init = decoder.__init__

    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        prefix = str(kwargs.get("prefix", ""))
        match = re.search(r"layers\.(\d+)", prefix)
        layer = int(match.group(1)) if match else 0
        calls = {"value": 0}

        def pre(_owner, inputs):
            _emit("layer_input", inputs[0], layer=layer, call=calls["value"])

        def post(_owner, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            _emit("layer_output", value, layer=layer, call=calls["value"])
            calls["value"] += 1

        self.register_forward_pre_hook(pre)
        self.register_forward_hook(post)
        for stage, owner in (
            ("qkv", self.self_attn.qkv_proj),
            ("oproj", self.self_attn.o_proj),
            ("router_logits", self.mlp.gate),
            ("moe_output", self.mlp),
        ):
            original = owner.forward

            def forward(target, *f_args, _stage=stage, _original=original, **f_kwargs):
                result = _original(*f_args, **f_kwargs)
                value = result[0] if isinstance(result, tuple) else result
                _emit(_stage, value, layer=layer, call=calls["value"])
                return result

            owner.forward = types.MethodType(forward, owner)

    decoder.__init__ = initialize


__all__ = ["install_stage_dump"]
