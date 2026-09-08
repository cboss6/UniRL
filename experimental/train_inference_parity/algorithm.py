"""Experimental algorithm wrapper that requires an explicit old-policy scorer."""

from __future__ import annotations

from unirl.algorithms.grpo import GRPO


class TrainInferenceParityGRPO(GRPO):
    """GRPO with a fail-closed full-sequence old-policy parity probe."""

    def __init__(
        self,
        *args,
        alignment_probe: bool = True,
        alignment_require_exact: bool = True,
        **kwargs,
    ) -> None:
        if not alignment_probe or not alignment_require_exact:
            raise ValueError("TrainInferenceParityGRPO requires alignment_probe=true and alignment_require_exact=true")
        super().__init__(
            *args,
            alignment_probe=True,
            alignment_require_exact=True,
            **kwargs,
        )

    def prepare_segment(self, *, conditions, segment) -> None:
        scorer = getattr(self.stage, "old_policy_replay", None)
        if not callable(scorer):
            raise RuntimeError("train-inference parity requires stage.old_policy_replay()")
        super().prepare_segment(conditions=conditions, segment=segment)


__all__ = ["TrainInferenceParityGRPO"]
