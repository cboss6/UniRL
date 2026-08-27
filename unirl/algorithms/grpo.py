"""Stage-driven ``GRPO`` over a ``TextSegment``."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Type

import torch

from unirl.types.conditions import Condition
from unirl.types.segments.text import TextSegment

from .base import (
    AlgorithmStepResult,
    BaseAlgorithmConfig,
    StageAlgorithm,
    _grpo_clip_loss,
    _resolve_clip_range_from_schedule,
    rollout_replay_k3,
    rollout_replay_logp_absdiff,
    typed_conditions,
)


@dataclass
class GRPOConfig(BaseAlgorithmConfig):
    stage_attr: str = "ar"
    conditions_cls: str = ""
    clip_range: float = 1e-4
    clip_schedule: str = "constant"
    alignment_probe: bool = False
    alignment_gate_only: bool = False


class GRPO(StageAlgorithm):
    """GRPO over an AR ``TextSegment`` via ``ARStage.replay``."""

    supports_multi_update = True

    def __init__(
        self,
        *,
        stage: Any = None,
        pipeline: Any = None,
        stage_attr: str = "ar",
        clip_range: float = 1e-4,
        clip_schedule: str = "constant",
        clip_range_high: Optional[float] = None,
        loss_agg_mode: str = "token-mean",
        horizon: int = 8192,
        conditions_cls: Optional[Type[Any]] = None,
        sampling_temperature: Optional[float] = None,
        alignment_probe: bool = False,
        alignment_gate_only: bool = False,
    ) -> None:
        super().__init__()
        if stage is None and pipeline is None:
            raise ValueError("GRPO: either `stage` or `pipeline` must be provided")
        if stage is None:
            stage = getattr(pipeline, stage_attr)
        self.stage = stage
        self.clip_range = float(clip_range)
        self.clip_range_high = None if clip_range_high is None else float(clip_range_high)
        self.clip_schedule = str(clip_schedule)
        self.loss_agg_mode = str(loss_agg_mode)
        self.horizon = int(horizon)
        self.conditions_cls = conditions_cls
        if sampling_temperature is None:
            from unirl.types.sampling import ARSamplingParams

            sampling_temperature = ARSamplingParams.__dataclass_fields__["temperature"].default
        self.sampling_temperature = float(sampling_temperature)
        self.alignment_probe = bool(alignment_probe)
        self.alignment_gate_only = bool(alignment_gate_only)

    def prepare_segment(
        self,
        *,
        conditions: Mapping[str, Condition],
        segment: "TextSegment",
    ) -> None:
        """Capture a no-grad actor replay separately from the rollout anchor."""
        if (
            not self.alignment_probe
            or segment.tokens is None
            or segment.log_probs is None
            or int(segment.tokens.shape[0]) == 0
        ):
            return
        if segment.rollout_log_probs is None:
            segment.rollout_log_probs = segment.log_probs.detach().cpu().clone()
        typed_conds = typed_conditions(conditions, self.conditions_cls)
        previous_record = os.environ.get("UNIMATCH_STAGE_DUMP_RECORD")
        if os.environ.get("UNIMATCH_STAGE_DUMP_DIR"):
            os.environ["UNIMATCH_STAGE_DUMP_RECORD"] = "batch"
        try:
            with torch.no_grad():
                actor_logp = self.stage.replay(
                    typed_conds,
                    segment=segment,
                    temperature=self.sampling_temperature,
                )
        finally:
            if previous_record is None:
                os.environ.pop("UNIMATCH_STAGE_DUMP_RECORD", None)
            else:
                os.environ["UNIMATCH_STAGE_DUMP_RECORD"] = previous_record
        segment.actor_log_probs = actor_logp.detach().cpu()
        if os.environ.get("UNIRL_K3_DEBUG", "0") == "1":
            rollout = segment.rollout_log_probs
            count = min(32, int(actor_logp.numel()))
            rollout_device = rollout.to(
                dtype=actor_logp.dtype,
                device=actor_logp.device,
            )
            mismatch = torch.nonzero(
                actor_logp != rollout_device,
                as_tuple=False,
            ).reshape(-1)
            prompt = getattr(typed_conds, "prompt", None)
            prompt_ids = getattr(prompt, "input_ids", None)
            print(
                "[unirl.k3.debug] "
                f"prompt_shape={tuple(prompt_ids.shape) if prompt_ids is not None else None} "
                f"prompt_tail={prompt_ids.reshape(-1)[-8:].detach().cpu().tolist() if prompt_ids is not None else None} "
                f"mismatch_count={int(mismatch.numel())} "
                f"first_mismatch={int(mismatch[0]) if mismatch.numel() else None} "
                f"last_mismatch={int(mismatch[-1]) if mismatch.numel() else None} "
                f"tokens={segment.tokens[:count].detach().cpu().tolist()} "
                f"rollout={rollout[:count].detach().cpu().tolist()} "
                f"actor={actor_logp[:count].detach().cpu().tolist()}",
                flush=True,
            )

    def compute_loss_and_backward(
        self,
        *,
        conditions: Mapping[str, Condition],
        segment: "TextSegment",
        advantages: torch.Tensor,
        training_progress: float,
        loss_scale: float,
    ) -> AlgorithmStepResult:
        if segment.tokens is None or segment.lengths is None or segment.log_probs is None:
            return AlgorithmStepResult(loss=0.0, metrics={}, num_steps_or_tokens=0, has_backward=False)
        if int(segment.tokens.shape[0]) == 0:
            return AlgorithmStepResult(loss=0.0, metrics={}, num_steps_or_tokens=0, has_backward=False)

        if self.alignment_gate_only:
            if segment.actor_log_probs is None:
                raise RuntimeError(
                    "GRPO alignment_gate_only requires alignment_probe actor logprobs"
                )
            rollout_logp = (
                segment.rollout_log_probs
                if segment.rollout_log_probs is not None
                else segment.log_probs
            )
            actor_logp = segment.actor_log_probs.to(
                dtype=rollout_logp.dtype,
                device=rollout_logp.device,
            )
            actor_absdiff = rollout_replay_logp_absdiff(actor_logp, rollout_logp)
            actor_k3 = rollout_replay_k3(actor_logp, rollout_logp)
            metrics = {
                "actor_rollout_logp_absdiff_mean": actor_absdiff[
                    "rollout_replay_logp_absdiff_mean"
                ],
                "actor_rollout_logp_absdiff_max": actor_absdiff[
                    "rollout_replay_logp_absdiff_max"
                ],
                "k3_mean": actor_k3["k3_mean"],
                "k3_max": actor_k3["k3_max"],
                **{
                    f"actor_rollout_{key}": value
                    for key, value in actor_k3.items()
                },
            }
            return AlgorithmStepResult(
                loss=0.0,
                metrics=metrics,
                num_steps_or_tokens=int(actor_logp.shape[0]),
                has_backward=False,
            )

        typed_conds = typed_conditions(conditions, self.conditions_cls)
        new_logp = self.stage.replay(typed_conds, segment=segment, temperature=self.sampling_temperature)
        old_logp = segment.log_probs.to(dtype=new_logp.dtype, device=new_logp.device)
        adv_per_token = self._expand_advantages_to_tokens(
            advantages, segment.lengths, dtype=new_logp.dtype, device=new_logp.device
        )

        clip_range = _resolve_clip_range_from_schedule(self.clip_range, self.clip_schedule, training_progress)
        clip_high = (
            None
            if self.clip_range_high is None
            else _resolve_clip_range_from_schedule(self.clip_range_high, self.clip_schedule, training_progress)
        )
        loss_per_elem, ratio_metrics = _grpo_clip_loss(
            new_logp=new_logp,
            old_logp=old_logp,
            advantages=adv_per_token,
            clip_range=clip_range,
            clip_range_high=clip_high,
        )

        if self.loss_agg_mode in ("seq-mean-token-sum-norm", "seq-mean-token-mean") and segment.lengths is not None:
            parts = torch.split(loss_per_elem, segment.lengths.tolist())
            if self.loss_agg_mode == "seq-mean-token-sum-norm":
                loss = torch.stack([p.sum() for p in parts]).mean() / float(self.horizon)
            else:  # seq-mean-token-mean — guard 0-length responses (mean of empty = NaN)
                loss = torch.stack([p.mean() if p.numel() else p.new_zeros(()) for p in parts]).mean()
        else:
            loss = loss_per_elem.mean()
        (loss * loss_scale).backward()

        metrics: Dict[str, Any] = {
            "policy_loss": float(loss.detach().item()),
            "clip_range": float(clip_range),
            **rollout_replay_logp_absdiff(new_logp, old_logp),
            **{k: float(v.item()) for k, v in ratio_metrics.items()},
        }
        if segment.actor_log_probs is not None:
            rollout_logp = (
                segment.rollout_log_probs
                if segment.rollout_log_probs is not None
                else segment.log_probs
            ).to(dtype=new_logp.dtype, device=new_logp.device)
            actor_logp = segment.actor_log_probs.to(
                dtype=new_logp.dtype, device=new_logp.device
            )
            actor_absdiff = rollout_replay_logp_absdiff(actor_logp, rollout_logp)
            actor_k3 = rollout_replay_k3(actor_logp, rollout_logp)
            metrics.update(
                {
                    "actor_rollout_logp_absdiff_mean": actor_absdiff[
                        "rollout_replay_logp_absdiff_mean"
                    ],
                    "actor_rollout_logp_absdiff_max": actor_absdiff[
                        "rollout_replay_logp_absdiff_max"
                    ],
                    "k3_mean": actor_k3["k3_mean"],
                    "k3_max": actor_k3["k3_max"],
                    **{
                        f"actor_rollout_{key}": value
                        for key, value in actor_k3.items()
                    },
                }
            )
        return AlgorithmStepResult(
            loss=float(loss.detach().item()),
            metrics=metrics,
            num_steps_or_tokens=int(new_logp.shape[0]),
            has_backward=True,
        )

    @staticmethod
    def _expand_advantages_to_tokens(
        advantages: torch.Tensor,
        lengths: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Expand per-sample ``advantages [B]`` to per-token ``[total_tokens]``."""
        bs = int(advantages.shape[0])
        if int(lengths.shape[0]) != bs:
            raise ValueError(f"GRPO advantage expansion: advantages batch={bs} != lengths={int(lengths.shape[0])}")
        chunks: List[torch.Tensor] = []
        adv_cast = advantages.detach().to(dtype=dtype, device=device)
        for k in range(bs):
            n = int(lengths[k].item())
            if n > 0:
                chunks.append(adv_cast[k].expand(n))
        if not chunks:
            return torch.zeros(0, dtype=dtype, device=device)
        return torch.cat(chunks, dim=0)


__all__ = ["GRPO", "GRPOConfig"]
