"""Qwen3OmniBundle — weights + processor holder for the Qwen3-Omni **thinker**.

The Qwen3-Omni checkpoint bundles three sub-models (thinker / talker / code2wav).
For text/MCQA RL we train only the **thinker** — a decoder-only Qwen3-MoE LM with
embedded vision + audio encoders. The standalone
``Qwen3OmniMoeThinkerForConditionalGeneration`` has ``base_model_prefix="thinker"``,
so ``from_pretrained`` against the full checkpoint auto-strips the ``thinker.``
prefix and loads only the thinker (talker / code2wav are reported as expected
unexpected keys and never instantiated — ~8B params saved).

Mirrors :class:`unirl.models.qwen_vl.QwenVLBundle` (transformer + processor +
tokenizer, freeze the vision tower) with an added audio-tower freeze. Lifecycle
concerns (LoRA injection, FSDP wrapping, offload) live outside the bundle.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from unirl.models.types.bundle import Bundle
from unirl.utils.dtypes import parse_torch_dtype

from .config import Qwen3OmniPipelineConfig

logger = logging.getLogger(__name__)


class Qwen3OmniBundle(Bundle):
    """Qwen3-Omni thinker bundle: thinker transformer + processor + tokenizer."""

    def __init__(
        self,
        *,
        transformer: nn.Module,
        processor: Any,
        tokenizer: Any,
        dtype: torch.dtype,
        device: torch.device,
        pretrained_path: str,
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.processor = processor
        self.tokenizer = tokenizer
        self.dtype = dtype
        self.device = device
        self.pretrained_path = pretrained_path

    @classmethod
    def from_config(cls, config: Qwen3OmniPipelineConfig) -> "Qwen3OmniBundle":
        from transformers import AutoConfig, AutoProcessor
        from transformers.models.qwen3_omni_moe import Qwen3OmniMoeThinkerForConditionalGeneration

        path = config.pretrained_model_ckpt_path

        device = config.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(device, str):
            device = torch.device(device)

        dtype = parse_torch_dtype(config.model_precision, field_name="model_precision")

        # The full-model config nests the thinker's own config under
        # ``thinker_config``; the standalone Thinker class is built from that.
        full_cfg = AutoConfig.from_pretrained(path, trust_remote_code=bool(config.trust_remote_code))
        thinker_cfg = full_cfg.thinker_config

        if config.meta_init_transformer:
            # Not yet supported: the checkpoint stores thinker weights under a
            # ``thinker.`` prefix while the standalone Thinker params are
            # unprefixed, so the shared ``load_sharded`` (all-shards +
            # set_model_state_dict on model param names) would not match. A
            # meta-init path needs a strip-``thinker.`` + drop-talker/code2wav
            # remap before set_model_state_dict. Eager auto-strips via
            # base_model_prefix, so use that until per-rank OOM forces meta-init.
            raise NotImplementedError(
                "Qwen3OmniBundle: meta_init_transformer=True is not yet supported "
                "(needs a strip-'thinker.' key remap in the sharded loader). Use "
                "meta_init_transformer=False (eager from_pretrained auto-strips the "
                "prefix via base_model_prefix='thinker')."
            )

        load_kwargs = {}
        if getattr(config, "attn_implementation", None):
            load_kwargs["attn_implementation"] = str(config.attn_implementation)

        # Eager load: from_pretrained on the FULL checkpoint with the thinker's
        # own config. base_model_prefix="thinker" makes HF strip the "thinker."
        # prefix and load only the thinker; talker/code2wav shards are ignored as
        # expected-unexpected keys (verified: 0 meta params, real weights, no
        # transformers-5.x _is_hf_initialized crash on the eager path).
        transformer = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
            path,
            config=thinker_cfg,
            dtype=dtype,
            low_cpu_mem_usage=True,
            **load_kwargs,
        ).to(device)

        # Structural (sets requires_grad / checkpointing flags, no weight access).
        # In the Qwen3-Omni thinker the encoders are TOP-LEVEL siblings of the
        # text decoder: transformer.visual (vision), transformer.audio_tower
        # (audio), transformer.model (Qwen3-MoE text decoder), transformer.lm_head.
        # Freeze the two encoders so only the LLM decoder trains (LoRA).
        if config.freeze_vision_tower and hasattr(transformer, "visual"):
            transformer.visual.requires_grad_(False)
            logger.info("Froze thinker vision tower (%d params).", sum(1 for _ in transformer.visual.parameters()))
        if config.freeze_audio_tower and hasattr(transformer, "audio_tower"):
            transformer.audio_tower.requires_grad_(False)
            logger.info("Froze thinker audio tower (%d params).", sum(1 for _ in transformer.audio_tower.parameters()))

        if config.use_gradient_checkpointing:
            if hasattr(transformer, "gradient_checkpointing_enable"):
                transformer.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            else:
                logger.warning(
                    "Qwen3-Omni thinker %s does not expose gradient_checkpointing_enable; skipping.",
                    type(transformer).__name__,
                )

        # The Qwen3-Omni processor handles text + image + video + audio. Load it
        # from the checkpoint root (preprocessor_config.json + chat_template.json).
        processor = AutoProcessor.from_pretrained(
            path,
            trust_remote_code=bool(config.trust_remote_code),
        )
        tokenizer = getattr(processor, "tokenizer", None) or processor
        if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token

        return cls(
            transformer=transformer,
            processor=processor,
            tokenizer=tokenizer,
            dtype=dtype,
            device=device,
            pretrained_path=path,
        )


__all__ = ["Qwen3OmniBundle"]
