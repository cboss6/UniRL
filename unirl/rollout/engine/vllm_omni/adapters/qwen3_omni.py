"""Qwen3-Omni Thinker family: input/output sub-adapters + the modality class.

Single AR stage (``worker_type: ar``), TP=4/8, LoRA-hot-loadable:

- One AR-only stage with ``model_arch:
  Qwen3OmniMoeThinkerForConditionalGeneration`` and ``scheduler_cls:
  OmniARScheduler``, configured by
  ``stage_configs/qwen3_omni_thinker_only_rl.yaml``.
- The prompt wire shape is the vllm-omni AR ``generate`` entry's:
  ``prompt = {"prompt_token_ids": ids, "multi_modal_data": {"video": vids}}``.

Driver-side responsibilities absorbed here:

- Load ``Qwen3OmniMoeProcessor`` once at adapter construction; encode each
  ``(text, video)`` pair the SAME way :class:`Qwen3OmniChatTemplateStage`
  does trainside — including pyav-based fps sampling + the ``size`` /
  ``fps`` / ``do_sample_frames`` template kwargs — so the rollout and the
  replay teacher-force over token-for-token identical prompts.
- Pack per-sample decoded frames as ``(frames_tensor, video_metadata_dict)``
  tuples for vllm-omni's worker to re-materialize.

Every ``RolloutReq`` produces exactly one :class:`GenerateCall` carrying the
whole batch. AR sampling knobs (``temperature`` / ``top_p`` / ``top_k`` /
``max_tokens`` / ``logprobs``) ride ``StageSampling(kind="ar", ...)``, which
the seam materializes into ``vllm.SamplingParams``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from unirl.config.require import require
from unirl.rollout.engine.vllm_omni.adapters.base import ModelAdapter, register_adapter
from unirl.rollout.engine.vllm_omni.adapters.hi3 import Hi3TextOutputAdapter
from unirl.rollout.engine.vllm_omni.backends import (
    STAGE_KIND_AR,
    GenerateCall,
    OmniRawResult,
    StageSampling,
)
from unirl.rollout.engine.vllm_omni.utils import seed_from_sample_id, texts_from_req
from unirl.types.primitives import Videos
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp


# --------------------------------------------------------------------------- #
# Video decode helper — mirrors trainside chat_template._sample_video_frames_pyav
# --------------------------------------------------------------------------- #


def _sample_video_frames_pyav(path: str, target_fps: float) -> Any:
    """Decode ``path`` and fps-sample frames as ``[T, C, H, W]`` uint8.

    Mirror of
    ``unirl.models.qwen3_omni.chat_template._sample_video_frames_pyav``,
    kept as a local copy (rather than an import) so the rollout adapter has
    no dependency on the models package.
    """
    import av
    import numpy as np
    import torch

    container = av.open(path)
    try:
        stream = container.streams.video[0]
        src_fps = float(stream.average_rate) if stream.average_rate else target_fps
        step = max(1, round(src_fps / float(target_fps)))
        frames = [
            frame.to_ndarray(format="rgb24")
            for i, frame in enumerate(container.decode(video=0))
            if i % step == 0
        ]
    finally:
        container.close()
    if not frames:
        raise ValueError(f"pyav decoded no frames from video: {path}")
    arr = np.stack(frames)  # [T, H, W, C]
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()


# --------------------------------------------------------------------------- #
# Input adapter
# --------------------------------------------------------------------------- #


class Qwen3OmniThinkerInputAdapter:
    """``RolloutReq`` → one :class:`GenerateCall` for the AR stage.

    Loads a driver-side ``Qwen3OmniMoeProcessor`` at construction. Each
    ``(text, video)`` pair is chat-templated + tokenized with the SAME kwargs
    the trainside stage uses (``fps``, ``do_sample_frames=False``, optional
    ``size`` cap for per-frame pixels), then wrapped in the vllm-omni AR
    prompt dict: ``{"prompt_token_ids": ids, "multi_modal_data": {"video":
    [(frames, meta), ...]}}``.

    The output is one :class:`GenerateCall` carrying the whole batch (vllm-omni
    handles per-request grouping via the ``"{i}_{uuid}"`` request-id prefix).
    """

    def __init__(
        self,
        modality: str,
        *,
        model_path: str,
        video_fps: float = 1.0,
        video_max_pixels: Optional[int] = None,
        use_audio_in_video: bool = False,
        max_prompt_length: int = 12288,
        system_instruction: Optional[str] = None,
    ) -> None:
        self.modality = modality
        self.model_path = str(model_path)
        self.video_fps = float(video_fps)
        self.video_max_pixels = int(video_max_pixels) if video_max_pixels else None
        self.use_audio_in_video = bool(use_audio_in_video)
        self.max_prompt_length = int(max_prompt_length)
        self.system_instruction = system_instruction

        # Load the processor + tokenizer once. trust_remote_code=True mirrors
        # the trainside bundle load.
        from transformers import AutoProcessor, AutoTokenizer

        self._processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        # Some Qwen3-Omni tokenizer builds ship chat_template.json separately.
        if getattr(self._tokenizer, "chat_template", None) is None:
            import json
            import os

            path = os.path.join(self.model_path, "chat_template.json")
            if os.path.exists(path):
                try:
                    with open(path) as f:
                        data = json.load(f)
                        if data.get("chat_template"):
                            self._tokenizer.chat_template = data["chat_template"]
                except (OSError, json.JSONDecodeError):
                    pass

        # Per-sample processor encodings from the most-recent ``build()`` call,
        # used by :class:`Qwen3OmniThinkerOutputAdapter` to assemble the
        # replay ``Qwen3OmniARConditions`` (prompt input_ids/attention_mask +
        # video tensors). Trainside ``Qwen3OmniPipeline.generate`` produces the
        # SAME conditions from the SAME processor encoding — so the GSPO
        # replay teacher-forces over token-for-token identical prompts.
        self._last_encodings: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # Video extraction — uri → pyav-decoded frames; already-decoded passes through
    # ------------------------------------------------------------------ #

    def _extract_videos(self, req: RolloutReq, n: int) -> List[Optional[Any]]:
        """Return one entry per prompt: pyav-decoded frames tensor, or ``None``."""
        prim = req.primitives.get("video")
        if prim is None:
            return [None] * n
        if not isinstance(prim, Videos):
            raise TypeError(
                f"Qwen3OmniThinkerInputAdapter: req.primitives['video'] must be Videos, "
                f"got {type(prim).__name__}"
            )
        # Prefer uris (raw paths) — the processor's TMRoPE derivation is
        # decode-step-independent as long as we sample at ``self.video_fps``.
        uris = getattr(prim, "uris", None)
        if uris:
            require(
                len(uris) == n,
                f"Qwen3OmniThinkerInputAdapter: uris count {len(uris)} != prompt count {n}",
            )
            return [_sample_video_frames_pyav(u, self.video_fps) for u in uris]
        # Fallback: caller passed pre-decoded packed frames — unpack via
        # cu_frames boundaries.
        frames = prim.frames
        cu = prim.cu_frames
        if frames is None or cu is None:
            raise ValueError(
                "Qwen3OmniThinkerInputAdapter: Videos primitive carries neither uris nor packed frames."
            )
        cu_list = [int(x) for x in cu.tolist()]
        require(
            len(cu_list) - 1 == n,
            f"Qwen3OmniThinkerInputAdapter: video batch {len(cu_list) - 1} != prompt count {n}",
        )
        return [frames[cu_list[i] : cu_list[i + 1]] for i in range(n)]

    # ------------------------------------------------------------------ #
    # Chat-template + processor encoding — mirrors Qwen3OmniChatTemplateStage.embed
    # ------------------------------------------------------------------ #

    def _encode_one(
        self,
        text: str,
        video_frames: Optional[Any],
        system_instruction: Optional[str],
    ) -> Dict[str, Any]:
        content: List[Dict[str, Any]] = []
        if video_frames is not None:
            content.append({"type": "video", "video": video_frames})
        content.append({"type": "text", "text": text})

        messages: List[Dict[str, Any]] = []
        if system_instruction is not None:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": content})

        template_kwargs: Dict[str, Any] = dict(
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        if video_frames is not None:
            template_kwargs["fps"] = self.video_fps
            template_kwargs["do_sample_frames"] = False
            if self.video_max_pixels is not None:
                template_kwargs["size"] = {
                    "shortest_edge": int(self._processor.video_processor.size["shortest_edge"]),
                    "longest_edge": self.video_max_pixels,
                }
            if self.use_audio_in_video:
                template_kwargs["use_audio_in_video"] = True
        return self._processor.apply_chat_template(messages, **template_kwargs)

    # ------------------------------------------------------------------ #
    # build_inputs
    # ------------------------------------------------------------------ #

    def build(self, req: RolloutReq) -> List[GenerateCall]:
        texts = texts_from_req(req)
        n = len(texts.texts)
        video_frames = self._extract_videos(req, n)

        # Per-request system instruction override (matches trainside behavior).
        chat_overrides = dict(req.stage_config.get("chat") or {})
        sys_instr = chat_overrides.get("system_instruction", self.system_instruction)

        prompts: List[Dict[str, Any]] = []
        # Reset the encoding cache — the output adapter reads it right after
        # ``rollout.generate`` returns to assemble replay conditions.
        self._last_encodings = []
        for text, vf in zip(texts.texts, video_frames):
            enc = self._encode_one(text, vf, sys_instr)
            ids = enc["input_ids"].squeeze(0).tolist()
            if len(ids) > self.max_prompt_length:
                # Multimodal placeholders align 1:1 with video features — a
                # truncation would corrupt them. Fail loudly.
                if vf is not None:
                    raise ValueError(
                        f"Qwen3OmniThinkerInputAdapter: multimodal prompt produced {len(ids)} tokens, "
                        f"exceeding max_prompt_length={self.max_prompt_length}. Reduce video_max_pixels "
                        "or video_fps, or raise max_prompt_length."
                    )
                ids = ids[-self.max_prompt_length :]
            entry: Dict[str, Any] = {"prompt_token_ids": ids}
            if vf is not None:
                # The vllm-omni AR server unpacks {"video": [...]} into
                # multi_modal_data.
                entry["multi_modal_data"] = {"video": [vf]}
            prompts.append(entry)
            # Cache the FULL encoding (input_ids, attention_mask, video tensors)
            # for the output adapter — trainside uses the SAME processor call
            # to build replay conditions, so caching here gives token-for-token
            # parity with trainside.
            self._last_encodings.append(enc)

        # AR sampling knobs — read from ``sampling_params["ar"]`` (v2 canonical).
        ar = req.sampling_params.get("ar")
        max_new_tokens = int(getattr(ar, "max_new_tokens", 512))
        temperature = float(getattr(ar, "temperature", 1.0))
        top_p = float(getattr(ar, "top_p", 1.0))
        top_k_val = int(getattr(ar, "top_k", 0))
        top_k = top_k_val if top_k_val > 0 else -1  # vLLM: -1 disables top_k
        stop_token_id = getattr(ar, "stop_token_id", None)

        sampling_kwargs: Dict[str, Any] = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_tokens": max_new_tokens,
            "logprobs": 1,  # per-token logprob on the sampled token
            # Deterministic per-sample seed so a GRPO group's N requests
            # don't collapse (patch_per_request_ar_seed in the runtime bundle
            # forwards this).
            "seed": seed_from_sample_id(req.sample_ids[0]) if req.sample_ids else 0,
        }
        if stop_token_id is not None:
            sampling_kwargs["stop_token_ids"] = [int(stop_token_id)]

        return [
            GenerateCall(
                prompts=prompts,
                sampling=[StageSampling(kind=STAGE_KIND_AR, kwargs=sampling_kwargs)],
            )
        ]


# --------------------------------------------------------------------------- #
# Output adapter
# --------------------------------------------------------------------------- #


class Qwen3OmniThinkerOutputAdapter(Hi3TextOutputAdapter):
    """AR-track response + replay conditions assembled from the driver-side
    processor encoding cached by :class:`Qwen3OmniThinkerInputAdapter`.

    Segments/decoded reuse :class:`Hi3TextOutputAdapter` verbatim (AR token
    packing is family-agnostic). ``build_conditions`` overrides HI3's empty
    default and populates a :class:`Qwen3OmniARConditions.to_dict()` payload —
    the SAME condition shape that trainside ``Qwen3OmniPipeline.generate``
    produces from the SAME processor call, so GSPO replay teacher-forces over
    token-for-token identical prompts.

    Requires the paired ``Qwen3OmniThinkerInputAdapter`` instance so we can
    read ``_last_encodings`` — the per-sample dicts from
    ``processor.apply_chat_template(..., return_dict=True)``.
    """

    def __init__(self, modality: str, input_adapter: "Qwen3OmniThinkerInputAdapter") -> None:
        super().__init__(modality)
        self._input_adapter = input_adapter

    def build_conditions(self, req: RolloutReq, per_request: List[List[OmniRawResult]]) -> Dict[str, Any]:
        """Assemble ``Qwen3OmniARConditions.to_dict()`` from cached encodings.

        Cached encodings are one dict per sample (index-aligned with
        ``req.sample_ids``), keys are the ``AutoProcessor`` outputs:

            input_ids            [1, L_i]  int64
            attention_mask       [1, L_i]  int64/float
            pixel_values_videos  [T_i * H*W, feat] (optional; ragged along T)
            video_grid_thw       [1, 3]           (optional)
            video_second_per_grid [1]             (optional)

        We right-pad ``input_ids`` / ``attention_mask`` to the batch-max L
        (pad-id for input_ids, 0 for mask), matching trainside's
        :meth:`unirl.models.qwen3_omni.chat_template.Qwen3OmniChatTemplateStage.embed`
        packing (which produces ``[B, max_L]``). Video tensors ride as CONCAT
        per-sample lists (None entries preserved for text-only samples).
        """
        del req, per_request
        import torch

        from unirl.models.qwen3_omni.conditions import Qwen3OmniARConditions
        from unirl.types.conditions import TextTokenCondition

        encs = self._input_adapter._last_encodings
        if not encs:
            raise RuntimeError(
                "Qwen3OmniThinkerOutputAdapter.build_conditions: input adapter "
                "cache is empty — ``build_inputs`` must run before ``build_response``."
            )

        pad_id = self._input_adapter._tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self._input_adapter._tokenizer.eos_token_id or 0

        # Squeeze the [1, L] leading dim and pad to batch max L.
        raw_ids = [e["input_ids"].squeeze(0) for e in encs]
        raw_masks = [e["attention_mask"].squeeze(0) for e in encs]
        max_len = int(max(t.shape[-1] for t in raw_ids))
        batch = len(raw_ids)
        input_ids = torch.full((batch, max_len), int(pad_id), dtype=torch.long)
        attention_mask = torch.zeros((batch, max_len), dtype=raw_masks[0].dtype if raw_masks else torch.long)
        for i, (ids, mask) in enumerate(zip(raw_ids, raw_masks)):
            L = int(ids.shape[-1])
            input_ids[i, :L] = ids[:L].to(torch.long)
            attention_mask[i, :L] = mask[:L]

        # Videos are per-sample optional tensors (CONCAT list in the Batch).
        pixel_values_videos: List[Any] = []
        video_grid_thw: List[Any] = []
        video_second_per_grid: List[Any] = []
        for e in encs:
            pvv = e.get("pixel_values_videos")
            vgt = e.get("video_grid_thw")
            vspg = e.get("video_second_per_grid")
            pixel_values_videos.append(pvv if pvv is not None else None)
            video_grid_thw.append(vgt if vgt is not None else None)
            if vspg is not None and not isinstance(vspg, torch.Tensor):
                vspg = torch.as_tensor(vspg)
            video_second_per_grid.append(vspg if vspg is not None else None)

        has_video = any(p is not None for p in pixel_values_videos)
        cond = Qwen3OmniARConditions(
            prompt=TextTokenCondition(input_ids=input_ids, attention_mask=attention_mask),
            pixel_values_videos=pixel_values_videos if has_video else None,
            video_grid_thw=video_grid_thw if has_video else None,
            video_second_per_grid=video_second_per_grid if has_video else None,
        )
        return cond.to_dict()


# --------------------------------------------------------------------------- #
# Modality binder
# --------------------------------------------------------------------------- #


@register_adapter("qwen3_omni_thinker")
class Qwen3OmniThinkerAdapter(ModelAdapter):
    """Qwen3-Omni Thinker — text/video → AR text (single stage, TP>1, LoRA)."""

    # ---- topology knobs ----
    stage_yaml = "qwen3_omni_thinker_only_rl.yaml"
    stage_yaml_source = "local"
    omni_mode = None  # AR-only; do NOT pass mode=text-to-image
    needs_sigmas = False  # AR path never reads sigmas
    # Adapter builds its own processor + tokenizer; the seam-side tokenize_prompt
    # (HI3-specific) is never invoked.
    needs_driver_tokenizer = False
    # The recipe uses LoRA; propagate the adapter through the ``Omni.generate``
    # ``lora_request`` kwarg (requires patch_lora_request_passthrough).
    ar_lora_passthrough = True
    # TP>1 stage — pop the Ray-injected ``CUDA_VISIBLE_DEVICES`` before booting
    # vllm-omni so the engine sees ALL host GPUs and can pin its TP workers per
    # ``runtime.devices`` in the stage YAML. Set True because the intended trainer
    # anchors this engine to a single Worker actor (``ARTrainer`` with
    # ``rollout_anchor_device`` set); SPMD callers that scatter-instantiate the
    # engine per-rank must NOT set this — they intentionally see one card apiece.
    clear_cuda_visible = True
    # TP>1 stage — wake-time LoRA re-push must use the byte-copy transport
    # (zero-copy handle crashes ranks 2..N).
    lora_copy_transport = True

    def __init__(
        self,
        config: Any,
        model_config: Any,
        *,
        strategy: Any = None,
        tokenize_fn: Any = None,
    ) -> None:
        # Skip the base ``validate()``'s shift path via needs_sigmas=False.
        # No σ schedule for AR-only.
        super().__init__(config, model_config, strategy=strategy, tokenize_fn=tokenize_fn)

        # Pull adapter-facing knobs off the model_config (the training-side
        # ``Qwen3OmniPipelineConfig``). Fall back to safe defaults when the
        # engine is constructed without a model config.
        mc = model_config
        model_path = str(config.model_path)
        video_fps = float(getattr(mc, "video_fps", 1.0)) if mc is not None else 1.0
        video_max_pixels = getattr(mc, "video_max_pixels", None) if mc is not None else None
        use_audio_in_video = bool(getattr(mc, "use_audio_in_video", False)) if mc is not None else False
        max_prompt_length = int(getattr(mc, "max_prompt_length", 12288)) if mc is not None else 12288
        system_instruction = getattr(mc, "system_instruction", None) if mc is not None else None

        self.input_adapter = Qwen3OmniThinkerInputAdapter(
            self.modality,
            model_path=model_path,
            video_fps=video_fps,
            video_max_pixels=video_max_pixels,
            use_audio_in_video=use_audio_in_video,
            max_prompt_length=max_prompt_length,
            system_instruction=system_instruction,
        )
        self.output_adapter = Qwen3OmniThinkerOutputAdapter(self.modality, self.input_adapter)

    # ---- σ schedule opt-out (base builds a FlowMatch policy from mc.shift) ----
    def schedule_policy(self) -> Any:
        """AR-only — no σ schedule; the engine's ``ensure_req_sigmas`` gate
        (``needs_sigmas=False``) never consults this.
        """
        return None

    # ---- request gating ----
    def validate_request(self, req: RolloutReq) -> None:
        if req.primitives.get("image") is not None:
            raise ValueError(
                f"modality={self.modality!r} rejects image-bearing requests; "
                "use req.primitives['video'] for multimodal input."
            )

    # ---- delegation ----
    def build_inputs(self, req: RolloutReq) -> List[GenerateCall]:
        return self.input_adapter.build(req)

    def build_response(self, req: RolloutReq, per_request: List[List[OmniRawResult]]) -> RolloutResp:
        return self.output_adapter.build(req, per_request)


__all__ = ["Qwen3OmniThinkerAdapter", "Qwen3OmniThinkerInputAdapter"]
