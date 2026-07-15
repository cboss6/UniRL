"""Qwen3-Omni Thinker model-specific patches — idempotent, applied at boot.

Registers the Thinker with ``AutoModelForCausalLM`` and applies the FSDP-init,
processor, and tokenizer fixups the Qwen3-Omni thinker needs. The four steps
this module makes are:

1. Register ``Qwen3OmniMoeForConditionalGeneration`` with
   ``AutoModelForCausalLM``, delegate ``forward`` / embeddings to
   ``self.thinker``, correct ``_no_split_modules`` and declare
   ``_verl_strip_modules`` so FSDP drops talker / code2wav / code_predictor.
2. Force ``Qwen3OmniMoeConfig.tie_word_embeddings=False`` via a descriptor
   — tied embeddings disable FSDP's meta-tensor init path and OOM on 30B-A3B.
3. Extend ``verl.utils.tokenizer.hf_processor`` to recognize
   ``Qwen3OmniMoeProcessor`` (Qwen3-Omni ships a multimodal processor that
   upstream's ``match`` block doesn't cover, so we wrap it).
4. Extend ``verl.utils.tokenizer.hf_tokenizer`` to auto-load
   ``chat_template.json`` when it isn't already on the tokenizer.

Registration model: this module is imported (a) inside vllm-omni's spawned
workers via :func:`unirl.rollout.engine.vllm_omni.patches.install`, and (b)
optionally on the driver via :func:`apply_on_driver` (so trainside code that
loads the model via ``AutoModelForCausalLM`` sees the same registration).

All functions are guarded and can be called repeatedly without side effects.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_APPLIED = False


def _register_qwen3_omni_automodel() -> None:
    """Register the Thinker with AutoModelForCausalLM and patch FSDP-init blockers.

    verl's ``_architecture_to_auto_class`` is optional here (only used by
    verl's FSDP engine); if the import fails we silently skip that map —
    the ``AutoModelForCausalLM.register`` call still runs.
    """
    try:
        from transformers import AutoModelForCausalLM
        from transformers.models.qwen3_omni_moe import (
            Qwen3OmniMoeConfig,
            Qwen3OmniMoeForConditionalGeneration,
        )
    except ImportError:
        return

    # verl's architecture map — optional on UniRL (only used when the model is
    # loaded via verl.utils.model.get_model). Skip cleanly if verl isn't around.
    try:
        from verl.utils.model import _architecture_to_auto_class

        _architecture_to_auto_class.setdefault(
            "Qwen3OmniMoeForConditionalGeneration", AutoModelForCausalLM
        )
    except ImportError:
        pass

    def _qwen3_omni_get_input_embeddings(self: Any) -> Any:
        return self.thinker.get_input_embeddings()

    def _qwen3_omni_set_input_embeddings(self: Any, value: Any) -> None:
        self.thinker.set_input_embeddings(value)

    def _qwen3_omni_forward(
        self: Any,
        input_ids: Any = None,
        attention_mask: Any = None,
        position_ids: Any = None,
        past_key_values: Any = None,
        inputs_embeds: Any = None,
        labels: Any = None,
        use_cache: Any = None,
        output_attentions: Any = None,
        output_hidden_states: Any = None,
        return_dict: Any = None,
        **kwargs: Any,
    ) -> Any:
        return self.thinker(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

    Qwen3OmniMoeForConditionalGeneration.forward = _qwen3_omni_forward
    Qwen3OmniMoeForConditionalGeneration.get_input_embeddings = _qwen3_omni_get_input_embeddings
    Qwen3OmniMoeForConditionalGeneration.set_input_embeddings = _qwen3_omni_set_input_embeddings
    # Upstream lists Qwen3OmniMoeDecoderLayer which does not exist; fix to
    # the real class name.
    Qwen3OmniMoeForConditionalGeneration._no_split_modules = ["Qwen3OmniMoeThinkerTextDecoderLayer"]
    # Read by verl's FSDPEngine — talker/code2wav/code_predictor aren't needed
    # for Thinker-only training. Left as an attribute regardless of whether
    # verl is imported (harmless if unused).
    Qwen3OmniMoeForConditionalGeneration._verl_strip_modules = [
        "talker",
        "code2wav",
        "code_predictor",
    ]

    # tie_word_embeddings=True forces use_meta_tensor=False during FSDP init,
    # which OOMs on 30B-A3B. Pin it to False at the class level with a
    # descriptor so subsequent ``config.tie_word_embeddings = ...`` assignments
    # in ``__init__`` become no-ops.
    logger.warning(
        "unirl.vllm_omni.patches.qwen3_omni: forcing tie_word_embeddings=False on "
        "Qwen3OmniMoeConfig — tied embeddings disable the FSDP meta-tensor init "
        "path and OOM on 30B-A3B."
    )

    class _FalseTieDescriptor:
        def __get__(self, obj: Any, objtype: Any = None) -> bool:
            return False

        def __set__(self, obj: Any, value: Any) -> None:
            pass

    Qwen3OmniMoeConfig.tie_word_embeddings = _FalseTieDescriptor()
    AutoModelForCausalLM.register(Qwen3OmniMoeConfig, Qwen3OmniMoeForConditionalGeneration)


def _patch_hf_processor_for_qwen3_omni() -> None:
    """Wrap ``verl.utils.tokenizer.hf_processor`` to recognize Qwen3OmniMoeProcessor.

    The original uses a ``match`` block that cannot be extended at runtime; we
    install a wrapper that handles the Qwen3-Omni case only when the original
    returns ``None`` (so other models are unaffected).

    Silently skips if verl isn't importable — this patch only matters for
    codepaths that reach ``verl.utils.tokenizer.hf_processor``, which is
    verl-internal; UniRL rollout adapters load the processor directly.
    """
    try:
        from transformers.models.qwen3_omni_moe import Qwen3OmniMoeThinkerForConditionalGeneration
    except ImportError:
        return

    try:
        import verl.utils.tokenizer as _vt
    except ImportError:
        return

    import types

    _original_hf_processor = _vt.hf_processor

    def _patched_hf_processor(name_or_path: Any, **kwargs: Any) -> Any:
        result = _original_hf_processor(name_or_path, **kwargs)
        if result is not None:
            return result

        # Original returned None — either tokenizer-only (fine) or Qwen3-Omni
        # tripped the unsupported-processor path.
        try:
            from transformers import AutoConfig, AutoProcessor, PreTrainedTokenizerBase

            processor = AutoProcessor.from_pretrained(name_or_path, **kwargs)
            if isinstance(processor, PreTrainedTokenizerBase):
                return None
            if processor.__class__.__name__ != "Qwen3OmniMoeProcessor":
                return None

            config = AutoConfig.from_pretrained(name_or_path, **kwargs)
            # Token IDs / spatial_merge_size live on ``thinker_config``, not
            # the top-level ``Qwen3OmniMoeConfig`` that AutoConfig returns.
            processor.config = config.thinker_config
            processor.spatial_merge_size = config.thinker_config.vision_config.spatial_merge_size
            model_class = Qwen3OmniMoeThinkerForConditionalGeneration
            processor.get_rope_index = types.MethodType(model_class.get_rope_index, processor)
            processor.get_llm_pos_ids_for_vision = types.MethodType(
                model_class.get_llm_pos_ids_for_vision, processor
            )
            return processor
        except Exception:
            return None

    _vt.hf_processor = _patched_hf_processor
    # Refresh verl.utils's stale re-export (callers use ``from verl.utils import
    # hf_processor``).
    import sys as _sys

    _utils_mod = _sys.modules.get("verl.utils")
    if _utils_mod is not None and hasattr(_utils_mod, "hf_processor"):
        _utils_mod.hf_processor = _patched_hf_processor


def _patch_hf_tokenizer_for_qwen3_omni() -> None:
    """Wrap ``verl.utils.tokenizer.hf_tokenizer`` to auto-load chat_template.json.

    Some models (Qwen3-Omni included) store the chat template in a separate
    ``chat_template.json`` instead of ``tokenizer_config.json``. This patch
    fills in ``tokenizer.chat_template`` when it's missing after load.

    Silent no-op if verl isn't importable.
    """
    import functools
    import json
    import os

    try:
        import verl.utils.tokenizer as _vt
    except ImportError:
        return

    _original_hf_tokenizer = _vt.hf_tokenizer

    @functools.wraps(_original_hf_tokenizer)
    def _patched_hf_tokenizer(name_or_path: Any, *args: Any, **kwargs: Any) -> Any:
        tokenizer = _original_hf_tokenizer(name_or_path, *args, **kwargs)

        if getattr(tokenizer, "chat_template", None) is None and isinstance(name_or_path, str):
            chat_template_path = os.path.join(name_or_path, "chat_template.json")
            if os.path.exists(chat_template_path):
                try:
                    with open(chat_template_path) as f:
                        data = json.load(f)
                        chat_template = data.get("chat_template")
                        if chat_template:
                            tokenizer.chat_template = chat_template
                except (OSError, json.JSONDecodeError):
                    pass

        return tokenizer

    _vt.hf_tokenizer = _patched_hf_tokenizer

    # Refresh already-imported ``hf_tokenizer`` bindings across verl modules.
    import sys

    for mod_name in list(sys.modules.keys()):
        if not mod_name.startswith("verl"):
            continue
        mod = sys.modules.get(mod_name)
        if (
            mod is not None
            and hasattr(mod, "hf_tokenizer")
            and mod.__dict__.get("hf_tokenizer") is _original_hf_tokenizer
        ):
            mod.hf_tokenizer = _patched_hf_tokenizer


def apply() -> None:
    """Apply all Qwen3-Omni Thinker patches (idempotent).

    Called from :func:`unirl.rollout.engine.vllm_omni.patches.install` at the
    spawn boundary; workers re-run it defensively via the mp.Process wrap in
    :mod:`unirl.rollout.engine.vllm_omni.patches.runtime`. Safe to call more
    than once — module-level ``_APPLIED`` gates repeat work.
    """
    global _APPLIED
    if _APPLIED:
        return
    _register_qwen3_omni_automodel()
    _patch_hf_processor_for_qwen3_omni()
    _patch_hf_tokenizer_for_qwen3_omni()
    _APPLIED = True


def apply_on_driver() -> None:
    """Driver-side entry point — same body as :func:`apply`.

    Exposed as a separate name so trainside code (``unirl.models.qwen3_omni``)
    can invoke the same registration before constructing the bundle, without
    growing a dependency on the rollout patches package's ``install()`` entry.
    """
    apply()


__all__ = ["apply", "apply_on_driver"]
