"""Unit tests for :mod:`unirl.rollout.coloc` (placement + colocate dance).

Zero real GPU / Ray dependency: everything mocks the engine handles and the
FSDP backend, and asserts the SEQUENCE of calls the scheduler makes. Runs as a
plain ``python -m unittest`` (or ``pytest``) — no fixtures beyond stdlib.

The dance we assert:

    train_step (sync=True):  extract → offload → wake → push → generate →
                             hydrate → sleep → onload
    train_step (sync=False): offload → wake → generate → hydrate → sleep → onload
    eval flow (sync=True):   extract → offload → wake → push
                             → [generate → hydrate]*N
                             → sleep → onload

    (offload_base=False variant simply drops the offload/onload calls.)
"""

from __future__ import annotations

import unittest
from typing import Any, List
from unittest.mock import MagicMock

from unirl.rollout.coloc import (
    ColocatedRolloutScheduler,
    derive_placement,
)


# ---------------------------------------------------------------------------
# Placement derivation
# ---------------------------------------------------------------------------


class PlacementDerivationTests(unittest.TestCase):
    def test_single_replica_colocate(self):
        n, anchors = derive_placement(total_gpus=4, rollout_world_size=4)
        self.assertEqual(n, 1)
        self.assertEqual(anchors, [1])  # base+1 offset (avoid rank-0 deadlock)

    def test_two_replicas(self):
        n, anchors = derive_placement(total_gpus=8, rollout_world_size=4)
        self.assertEqual(n, 2)
        self.assertEqual(anchors, [1, 5])

    def test_spmd_case(self):
        n, anchors = derive_placement(total_gpus=4, rollout_world_size=1)
        # SPMD: base+1 clamps to 0 when rws==1 (offset must land inside slice).
        self.assertEqual(n, 4)
        self.assertEqual(anchors, [0, 1, 2, 3])

    def test_divisibility_check(self):
        with self.assertRaises(ValueError):
            derive_placement(total_gpus=6, rollout_world_size=4)

    def test_rws_must_be_positive(self):
        with self.assertRaises(ValueError):
            derive_placement(total_gpus=4, rollout_world_size=0)


# ---------------------------------------------------------------------------
# Colocate dance sequencing
# ---------------------------------------------------------------------------


class _CallLog:
    """Records ``(target, method)`` for every mocked call, in order."""

    def __init__(self) -> None:
        self.events: List[str] = []

    def record(self, target: str, method: str) -> Any:
        def _wrapped(*args: Any, **kwargs: Any) -> Any:
            self.events.append(f"{target}.{method}")
            return MagicMock()

        return _wrapped


def _mock_engine(log: _CallLog, name: str = "engine") -> Any:
    """A minimal engine handle with the lifecycle verbs used by the scheduler."""
    eng = MagicMock()
    eng.wake_up.side_effect = log.record(name, "wake_up")
    eng.sleep.side_effect = log.record(name, "sleep")
    # ``generate`` returns a plain dict (no TensorRef leaves) so
    # ``pytree_hydrate`` is a structural no-op — the sequence we care about
    # is unaffected by the response payload.
    eng.generate.side_effect = log.record(name, "generate")
    eng.generate.return_value = {"tracks": {}}
    return eng


def _mock_backend(log: _CallLog) -> Any:
    b = MagicMock()
    b.onload.side_effect = log.record("backend", "onload")
    b.offload.side_effect = log.record("backend", "offload")
    return b


def _mock_sync(log: _CallLog) -> Any:
    s = MagicMock()
    s.extract.side_effect = log.record("sync", "extract")
    s.push.side_effect = log.record("sync", "push")
    s.sync.side_effect = log.record("sync", "sync")
    return s


class TrainStepDanceTests(unittest.TestCase):
    def _make(self, *, offload_base: bool, with_sync: bool):
        log = _CallLog()
        engine = _mock_engine(log)
        backend = _mock_backend(log)
        sync = _mock_sync(log) if with_sync else None
        sched = ColocatedRolloutScheduler(
            engines=[engine],
            backend=backend,
            weight_sync=sync,
            offload_base=offload_base,
        )
        return sched, log

    def test_train_step_with_sync_colocate(self):
        sched, log = self._make(offload_base=True, with_sync=True)
        sched.train_step_generate(req={}, sync_weights=True, rollout_id=0)
        # extract → offload → wake → push → generate → sleep → onload
        self.assertEqual(
            log.events,
            [
                "sync.extract",
                "backend.offload",
                "engine.wake_up",
                "sync.push",
                "engine.generate",
                "engine.sleep",
                "backend.onload",
            ],
        )

    def test_train_step_no_sync_colocate(self):
        sched, log = self._make(offload_base=True, with_sync=True)
        sched.train_step_generate(req={}, sync_weights=False, rollout_id=1)
        # No sync → skip extract/push; offload still fires to vacate cards.
        self.assertEqual(
            log.events,
            [
                "backend.offload",
                "engine.wake_up",
                "engine.generate",
                "engine.sleep",
                "backend.onload",
            ],
        )

    def test_train_step_no_offload_disjoint_slabs(self):
        """When the engine lives on cards disjoint from the training FSDP,
        the colocate offload is unnecessary and must NOT fire."""
        sched, log = self._make(offload_base=False, with_sync=True)
        sched.train_step_generate(req={}, sync_weights=True, rollout_id=2)
        self.assertEqual(
            log.events,
            [
                "sync.extract",
                "engine.wake_up",
                "sync.push",
                "engine.generate",
                "engine.sleep",
            ],
        )
        # And onload does not fire either (never offloaded).
        self.assertNotIn("backend.onload", log.events)

    def test_train_step_no_handler(self):
        sched, log = self._make(offload_base=True, with_sync=False)
        sched.train_step_generate(req={}, sync_weights=True, rollout_id=3)
        # sync_weights=True but no handler → skip both extract and push.
        self.assertEqual(
            log.events,
            [
                "backend.offload",
                "engine.wake_up",
                "engine.generate",
                "engine.sleep",
                "backend.onload",
            ],
        )


class EvalDanceTests(unittest.TestCase):
    def test_enter_generate_exit_colocate(self):
        log = _CallLog()
        engine = _mock_engine(log)
        backend = _mock_backend(log)
        sync = _mock_sync(log)
        sched = ColocatedRolloutScheduler(
            engines=[engine], backend=backend, weight_sync=sync, offload_base=True,
        )
        sched.enter_eval(sync_weights=True)
        for _ in range(3):
            sched.generate_batch({})
        sched.exit_eval()
        self.assertEqual(
            log.events,
            [
                "sync.extract",
                "backend.offload",
                "engine.wake_up",
                "sync.push",
                "engine.generate",
                "engine.generate",
                "engine.generate",
                "engine.sleep",
                "backend.onload",
            ],
        )


class ConstructionAssertionTests(unittest.TestCase):
    def test_multi_engine_train_generate_raises(self):
        log = _CallLog()
        sched = ColocatedRolloutScheduler(
            engines=[_mock_engine(log, "a"), _mock_engine(log, "b")],
            backend=_mock_backend(log),
            weight_sync=None,
            offload_base=True,
        )
        with self.assertRaises(NotImplementedError):
            sched.train_step_generate(req={}, sync_weights=False, rollout_id=0)


if __name__ == "__main__":
    unittest.main()
