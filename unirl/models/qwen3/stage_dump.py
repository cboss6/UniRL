"""Forward hooks for Qwen3 actor stage comparisons."""

from __future__ import annotations

import os


def install_qwen3_stage_dump_hooks(transformer) -> None:
    if not os.environ.get("UNIMATCH_STAGE_DUMP_DIR"):
        return
    if getattr(transformer, "_unimatch_stage_dump_installed", False):
        return

    from unimatch.diagnostics.stage_dump import emit_stage, stage_dump_enabled

    side = "veomni" if type(transformer).__module__.startswith("veomni.") else "fsdp"

    def tensor(value):
        return value[0] if isinstance(value, tuple) else value

    def emit(owner, stage, value, layer):
        if not stage_dump_enabled():
            return
        counters = getattr(owner, "_unimatch_stage_calls", None)
        if counters is None:
            counters = {}
            owner._unimatch_stage_calls = counters
        call = int(counters.get(stage, 0))
        counters[stage] = call + 1
        emit_stage(
            side,
            stage,
            tensor(value),
            layer=layer,
            call=call,
        )

    def pre(stage, layer, index=0):
        def hook(owner, inputs):
            if len(inputs) > index:
                emit(owner, stage, inputs[index], layer)

        return hook

    def post(stage, layer):
        def hook(owner, inputs, output):
            del inputs
            emit(owner, stage, output, layer)

        return hook

    model = getattr(transformer, "model", None)
    layers = getattr(model, "layers", None)
    if layers is None:
        raise RuntimeError("Qwen3 stage dump could not locate model.layers")
    for layer_index, layer in enumerate(layers):
        layer.register_forward_pre_hook(
            pre("layer_input", layer_index)
        )
        layer.register_forward_hook(post("layer_output", layer_index))
        attention = layer.self_attn
        attention.register_forward_pre_hook(
            pre("attention_input", layer_index)
        )
        attention.register_forward_hook(
            post("attention_output", layer_index)
        )
        attention.q_proj.register_forward_hook(post("q_proj", layer_index))
        attention.k_proj.register_forward_hook(post("k_proj", layer_index))
        attention.v_proj.register_forward_hook(post("v_proj", layer_index))
        attention.q_norm.register_forward_hook(post("q_norm", layer_index))
        attention.k_norm.register_forward_hook(post("k_norm", layer_index))
        attention.o_proj.register_forward_pre_hook(
            pre("oproj_input", layer_index)
        )
        attention.o_proj.register_forward_hook(
            post("oproj_output", layer_index)
        )
        mlp = layer.mlp
        mlp.register_forward_pre_hook(pre("moe_input", layer_index))
        mlp.register_forward_hook(post("moe_output", layer_index))
        gate = getattr(mlp, "gate", None)
        if gate is not None:
            gate.register_forward_hook(post("router_logits", layer_index))
    transformer._unimatch_stage_dump_installed = True
    print(
        f"[unirl.stage_dump] installed actor hooks for {len(layers)} layers",
        flush=True,
    )


__all__ = ["install_qwen3_stage_dump_hooks"]
