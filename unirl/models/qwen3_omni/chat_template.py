"""Qwen3OmniChatTemplateStage — Texts (+ optional Videos) → Qwen3OmniARConditions.

Uses the ``Qwen3OmniMoeProcessor`` to build prompt ``input_ids`` / ``attention_mask``
and, when videos are supplied, the ``pixel_values_videos`` / ``video_grid_thw`` /
``video_second_per_grid`` tensors the thinker consumes via TMRoPE.

Mirrors :class:`unirl.models.qwen_vl.QwenVLChatTemplateStage` (per-sample
right-pad-to-batch-max, media stored as per-sample CONCAT lists). Pass
``videos=None`` for text-only conditions.
"""

from __future__ import annotations

from typing import Any, List, Optional

import torch

from unirl.types.conditions import TextTokenCondition
from unirl.types.primitives import Texts

from .bundle import Qwen3OmniBundle
from .conditions import Qwen3OmniARConditions


def _sample_video_frames_pyav(path: str, target_fps: float) -> "torch.Tensor":
    """Decode ``path`` and sample frames at ``target_fps`` → ``[T, C, H, W]`` uint8.

    Some torchvision builds (0.26+) dropped ``io.read_video`` and the Qwen3-Omni
    processor's ``fetch_videos`` only tries torchcodec/torchvision (never pyav),
    so handing the processor a file PATH fails. Instead we decode + fps-sample
    here with pyav (present in the env) and hand the processor pre-decoded frames
    with ``do_sample_frames=False``. The processor still derives
    ``video_grid_thw`` from the frame count and ``second_per_grid`` from
    ``temporal_patch_size / fps`` (pure arithmetic, decode-independent), so TMRoPE
    stays consistent as long as this sampling fps equals the ``fps`` kwarg.
    """
    import av
    import numpy as np

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
    arr = np.stack(frames)  # [T, H, W, C] uint8
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()  # [T, C, H, W]


class Qwen3OmniChatTemplateStage:
    def __init__(
        self,
        bundle: Qwen3OmniBundle,
        *,
        system_instruction: Optional[str] = None,
        max_prompt_length: int = 4096,
        pad_to_max_length: bool = False,
        video_fps: float = 1.0,
        video_max_pixels: Optional[int] = None,
        use_audio_in_video: bool = False,
    ) -> None:
        self.bundle = bundle
        self.system_instruction = system_instruction
        self.max_prompt_length = int(max_prompt_length)
        # When True, pad every prompt to a fixed ``max_prompt_length`` instead of
        # the per-batch dynamic max. Required by the DP trainer: shards from
        # different rollout workers are concatenated (dim 0) at merge time, so
        # input_ids/attention_mask must share one sequence length across shards.
        self.pad_to_max_length = bool(pad_to_max_length)
        self.video_fps = float(video_fps)
        self.video_max_pixels = int(video_max_pixels) if video_max_pixels else None
        self.use_audio_in_video = bool(use_audio_in_video)

    def embed(
        self,
        texts: Texts,
        videos: Optional[List[Optional[Any]]] = None,
    ) -> Qwen3OmniARConditions:
        """Build conditions from prompts and optional per-sample videos.

        ``videos[i]`` is a decoded video for sample ``i`` (or ``None``); pass
        ``videos=None`` for a text-only batch.
        """
        processor = self.bundle.processor
        device = self.bundle.device
        dtype = self.bundle.dtype
        batch_size = len(texts.texts)

        per_sample_inputs = []
        for i, text in enumerate(texts.texts):
            content: list = []
            sample_video = None
            if videos is not None and i < len(videos) and videos[i] is not None:
                raw_video = videos[i]
                # A str/path is decoded + fps-sampled here with pyav (the processor
                # cannot decode paths in this env — see _sample_video_frames_pyav);
                # already-decoded frames (tensor/ndarray) pass through unchanged.
                if isinstance(raw_video, str):
                    sample_video = _sample_video_frames_pyav(raw_video, self.video_fps)
                else:
                    sample_video = raw_video
                # The <video> placeholder is materialized by the processor from
                # the video kwarg; the message just declares the video slot.
                content.append({"type": "video", "video": sample_video})
            content.append({"type": "text", "text": text})

            messages: list = []
            if self.system_instruction is not None:
                messages.append({"role": "system", "content": self.system_instruction})
            messages.append({"role": "user", "content": content})

            template_kwargs: dict = dict(
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            if sample_video is not None:
                # Qwen3-Omni processor kwarg is ``fps`` (NOT ``video_fps`` — that
                # name is silently ignored and the processor then samples EVERY
                # frame, exploding the token count). We already fps-sampled the
                # frames with pyav, so do_sample_frames=False (else the processor
                # re-samples and demands VideoMetadata we don't have). second_per_grid
                # is still temporal_patch_size / fps, so pass the same fps.
                template_kwargs["fps"] = self.video_fps
                template_kwargs["do_sample_frames"] = False
                if self.video_max_pixels is not None:
                    # Cap per-frame pixels so the processor smart-resizes each
                    # frame to <= video_max_pixels, bounding the video-token
                    # count. Only a per-call ``size`` dict is honored here — the
                    # processor ignores min/max_pixels set at from_pretrained and
                    # the bare ``max_pixels`` kwarg (verified against
                    # Qwen2VLVideoProcessor). shortest_edge keeps the processor's
                    # min-pixels floor.
                    template_kwargs["size"] = {
                        "shortest_edge": int(processor.video_processor.size["shortest_edge"]),
                        "longest_edge": self.video_max_pixels,
                    }
                if self.use_audio_in_video:
                    template_kwargs["use_audio_in_video"] = True
            inputs = processor.apply_chat_template(messages, **template_kwargs)
            per_sample_inputs.append(inputs)

        if self.pad_to_max_length:
            max_len = self.max_prompt_length
        else:
            max_len = min(
                max(inp["input_ids"].shape[-1] for inp in per_sample_inputs),
                self.max_prompt_length,
            )
        pad_id = self.bundle.tokenizer.pad_token_id
        if pad_id is None:
            raise RuntimeError(
                "Qwen3OmniChatTemplateStage.embed: tokenizer has no pad_token_id; "
                "Qwen3OmniBundle.from_config sets pad_token=eos_token when absent."
            )

        input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
        for i, inp in enumerate(per_sample_inputs):
            ids = inp["input_ids"].squeeze(0)
            L = min(int(ids.shape[0]), max_len)
            input_ids[i, :L] = ids[:L].to(device)
            mask = inp["attention_mask"].squeeze(0)
            attention_mask[i, :L] = mask[:L].to(device)

        # Per-sample video tensors (CONCAT lists), None for text-only samples.
        pixel_values_videos: List[Optional[torch.Tensor]] = []
        video_grid_thw: List[Optional[torch.Tensor]] = []
        video_second_per_grid: List[Optional[torch.Tensor]] = []
        for inp in per_sample_inputs:
            pvv = inp.get("pixel_values_videos")
            vgt = inp.get("video_grid_thw")
            vspg = inp.get("video_second_per_grid")
            pixel_values_videos.append(pvv.to(device=device, dtype=dtype) if pvv is not None else None)
            video_grid_thw.append(vgt.to(device=device) if vgt is not None else None)
            if vspg is not None:
                vspg_t = vspg if isinstance(vspg, torch.Tensor) else torch.as_tensor(vspg)
                video_second_per_grid.append(vspg_t.to(device=device))
            else:
                video_second_per_grid.append(None)

        has_video = any(p is not None for p in pixel_values_videos)
        return Qwen3OmniARConditions(
            prompt=TextTokenCondition(input_ids=input_ids, attention_mask=attention_mask),
            pixel_values_videos=pixel_values_videos if has_video else None,
            video_grid_thw=video_grid_thw if has_video else None,
            video_second_per_grid=video_second_per_grid if has_video else None,
        )


__all__ = ["Qwen3OmniChatTemplateStage"]
