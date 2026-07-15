"""Construction config for the typed Qwen3-Omni thinker AR pipeline.

Sibling of :class:`unirl.models.qwen3.Qwen3PipelineConfig` (pure causal LM) and
:class:`unirl.models.qwen_vl.QwenVLPipelineConfig` (multimodal). Qwen3-Omni's
**thinker** is a decoder-only LM with embedded vision + audio encoders; we train
only the thinker's LLM decoder (LoRA), freezing the vision/audio towers — so this
config carries the VL-style ``freeze_vision_tower`` plus an ``freeze_audio_tower``.

Weights + precision knobs only; LoRA injection, FSDP wrapping, gradient
checkpointing, and offload control live in the recipe's ``backend`` block — the
bundle is weights + params only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from unirl.config.validation import validate_precision_type


@dataclass
class Qwen3OmniPipelineConfig:
    """Construction args for ``Qwen3OmniPipeline.from_config`` / ``.from_bundle``.

    ``device`` may be runtime-injected by the actor after compose; the other
    fields are set at compose time and read once during bundle/pipeline
    construction.
    """

    pretrained_model_ckpt_path: str
    tokenizer_ckpt_path: Optional[str] = None
    trust_remote_code: bool = True

    model_precision: Any = "bf16"
    # HF attention backend for the TRAIN-side thinker, set on from_pretrained — so
    # it is the model's GLOBAL backend and governs EVERY forward: replay
    # teacher-forcing AND the HF autoregress() decode loop. Qwen3-Omni's thinker
    # text model is a Qwen3-MoE decoder; like Qwen3 it supports 'flex_attention'
    # for fast packed-varlen replay (transformers builds a BlockMask from the
    # restarting position_ids). None = HF default (sdpa) -> dense replay.
    attn_implementation: Optional[str] = None
    device: Any = None

    autocast_precision: str = "bf16"
    logprob_precision: str = "fp32"

    use_gradient_checkpointing: bool = False

    # The trainable module is the thinker (a decoder-only LM despite its
    # ForConditionalGeneration suffix); its params are unprefixed at the model
    # root (model.layers.* / visual.* / audio_tower.*). Weight-sync key prefix
    # mirrors qwen_vl's "model." (the thinker's decoder lives under .model).
    weight_sync_param_name_prefix: str = "model."

    use_lora: bool = False
    lora_target_modules: Optional[List[str]] = None

    # Freeze the two encoders embedded in the thinker. For a text/MCQA policy we
    # train only the LLM decoder (LoRA); the vision + audio towers stay cold.
    freeze_vision_tower: bool = True
    freeze_audio_tower: bool = True

    max_prompt_length: int = 4096

    # Video preprocessing knobs. fps governs the temporal sampling the Qwen3-Omni
    # processor applies; lower fps → fewer video tokens (fps=2 gives prompt≈20k
    # tokens, so start low).
    video_fps: float = 1.0
    video_max_frames: Optional[int] = None
    # Per-frame pixel cap fed to the processor as size={"longest_edge": N}. The
    # Qwen3-Omni video processor smart-resizes each frame to <= N pixels, so N
    # bounds the video-token count regardless of source resolution (a 1080p clip
    # at fps=1 costs ~12960 tokens uncapped — overflowing max_prompt_length and
    # truncating the <video> placeholders, which then mismatch the visual
    # features). None = processor default (effectively unbounded). NOTE: the video
    # processor ignores min/max_pixels at from_pretrained and only honors a
    # per-call size dict, so this is threaded through apply_chat_template, not the
    # bundle (unlike qwen_vl's image min/max_pixels).
    video_max_pixels: Optional[int] = None
    # Whether to fuse the audio track from the video into the thinker (TMRoPE
    # audio path). Default False (video-only; audio params stay None).
    use_audio_in_video: bool = False

    # Meta-init the thinker (build on meta; backend loads weights after sharding)
    # instead of eager ``from_pretrained``. NOTE: unlike qwen3/qwen_vl, the
    # Qwen3-Omni checkpoint stores thinker weights under a ``thinker.`` prefix
    # while the standalone Thinker model params are unprefixed — so the plain
    # ``load_sharded`` cannot be used as-is (it would need a strip-``thinker.`` +
    # drop-talker/code2wav remap). Left False until a per-rank OOM forces it;
    # eager ``from_pretrained`` auto-strips the prefix via base_model_prefix.
    meta_init_transformer: bool = False

    system_instruction: Optional[str] = None

    def __post_init__(self) -> None:
        validate_precision_type(self.model_precision, field="Qwen3OmniPipelineConfig.model_precision")


__all__ = ["Qwen3OmniPipelineConfig"]
