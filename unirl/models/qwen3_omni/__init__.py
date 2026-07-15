"""Qwen3-Omni **thinker** AR pipeline on the typed stage/pipeline architecture.

Sibling of :mod:`unirl.models.qwen3` (pure causal LM) and
:mod:`unirl.models.qwen_vl` (multimodal). Trains only the Qwen3-Omni thinker's
LLM decoder (LoRA), freezing the embedded vision + audio encoders; conditions
may be text or video (video via TMRoPE).

Importing this package re-exports its bundle / pipeline / config classes;
recipes wire them by ``_target_`` dotpath.
"""

from unirl.models.qwen3_omni.ar import (
    Qwen3OmniARParams,
    Qwen3OmniARStage,
    Qwen3OmniARStep,
)
from unirl.models.qwen3_omni.bundle import Qwen3OmniBundle
from unirl.models.qwen3_omni.chat_template import Qwen3OmniChatTemplateStage
from unirl.models.qwen3_omni.conditions import Qwen3OmniARConditions
from unirl.models.qwen3_omni.config import Qwen3OmniPipelineConfig
from unirl.models.qwen3_omni.pipeline import Qwen3OmniPipeline

__all__ = [
    "Qwen3OmniARConditions",
    "Qwen3OmniARParams",
    "Qwen3OmniARStage",
    "Qwen3OmniARStep",
    "Qwen3OmniBundle",
    "Qwen3OmniChatTemplateStage",
    "Qwen3OmniPipeline",
    "Qwen3OmniPipelineConfig",
]
