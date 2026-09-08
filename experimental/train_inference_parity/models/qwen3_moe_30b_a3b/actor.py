"""Qwen3 pipeline wrapper that validates the experiment's frozen model contract."""

from __future__ import annotations

from unirl.models.qwen3.pipeline import Qwen3Pipeline

from .contract import validate_model_config
from .stage_dump import install_actor_stage_dump


class ParityQwen3Pipeline(Qwen3Pipeline):
    @classmethod
    def from_bundle(cls, bundle, **kwargs):
        validate_model_config(bundle.transformer.config)
        install_actor_stage_dump(bundle.transformer)
        return super().from_bundle(bundle, **kwargs)


__all__ = ["ParityQwen3Pipeline"]
