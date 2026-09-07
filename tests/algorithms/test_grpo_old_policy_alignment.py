from __future__ import annotations

import pytest
import torch

from unirl.algorithms.grpo import GRPO
from unirl.types.segments.text import TextSegment


class _OldPolicyStage:
    def __init__(self, output: torch.Tensor) -> None:
        self.output = output
        self.old_policy_calls = 0
        self.generic_replay_calls = 0

    def old_policy_replay(self, conditions, *, segment, temperature):
        del conditions, segment, temperature
        self.old_policy_calls += 1
        return self.output.clone()

    def replay(self, conditions, *, segment, temperature):
        del conditions, segment, temperature
        self.generic_replay_calls += 1
        raise AssertionError("alignment probe must use old_policy_replay")


def _segment(log_probs: torch.Tensor) -> TextSegment:
    return TextSegment.pack(
        tokens=[torch.arange(log_probs.numel(), dtype=torch.long)],
        log_probs=[log_probs.clone()],
    )


def test_alignment_probe_uses_full_sequence_old_policy_replay() -> None:
    rollout = torch.tensor([-1.0, -2.0], dtype=torch.float32)
    stage = _OldPolicyStage(rollout)
    algorithm = GRPO(
        stage=stage,
        alignment_probe=True,
        alignment_gate_only=True,
        alignment_require_exact=True,
    )
    segment = _segment(rollout)

    algorithm.prepare_segment(conditions={}, segment=segment)
    result = algorithm.compute_loss_and_backward(
        conditions={},
        segment=segment,
        advantages=torch.tensor([0.0]),
        training_progress=0.0,
        loss_scale=1.0,
    )

    assert stage.old_policy_calls == 1
    assert stage.generic_replay_calls == 0
    assert torch.equal(segment.old_log_probs, rollout)
    assert result.metrics["old_rollout_logp_absdiff_max"] == 0.0
    assert result.metrics["old_rollout_k3_mean"] == 0.0
    assert result.metrics["old_rollout_k3_max"] == 0.0


def test_alignment_probe_rejects_one_ulp_even_when_k3_rounds_to_zero() -> None:
    rollout = torch.tensor([-1.0], dtype=torch.float32)
    different = torch.nextafter(rollout, torch.zeros_like(rollout))
    stage = _OldPolicyStage(different)
    algorithm = GRPO(
        stage=stage,
        alignment_probe=True,
        alignment_gate_only=True,
        alignment_require_exact=True,
    )

    with pytest.raises(
        RuntimeError,
        match=r"mismatch_count=1.*max_absdiff_fp32=5\.960464477539063e-08",
    ):
        algorithm.prepare_segment(conditions={}, segment=_segment(rollout))
