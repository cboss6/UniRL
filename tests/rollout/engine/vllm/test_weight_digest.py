from types import SimpleNamespace

import torch

from unirl.rollout.engine.vllm.worker_extension import (
    UniRLWeightSyncExtension,
)
from unirl.rollout.engine.vllm.engine import VLLMRolloutEngine


class _Runner:
    def __init__(self, model):
        self._model = model

    def get_model(self):
        return self._model


def _extension(model):
    extension = UniRLWeightSyncExtension()
    extension.rank = 0
    extension.model_runner = _Runner(model)
    return extension


def test_weight_digest_is_stable_for_identical_models():
    first = torch.nn.Sequential(
        torch.nn.Linear(4, 3, bias=False),
        torch.nn.Linear(3, 2, bias=False),
    )
    second = torch.nn.Sequential(
        torch.nn.Linear(4, 3, bias=False),
        torch.nn.Linear(3, 2, bias=False),
    )
    second.load_state_dict(first.state_dict())

    left = _extension(first).unirl_weight_digest()
    right = _extension(second).unirl_weight_digest()

    assert left["parameters"] == right["parameters"] == 2
    assert left["sha256"] == right["sha256"]


def test_weight_digest_detects_sampled_weight_change():
    parameter = torch.nn.Parameter(torch.arange(8.0))
    model = SimpleNamespace(
        named_parameters=lambda: [("model.weight", parameter)]
    )
    before = _extension(model).unirl_weight_digest()["sha256"]
    parameter.data[-1] += 1
    after = _extension(model).unirl_weight_digest()["sha256"]

    assert before != after


def test_staged_wake_tracks_partial_and_full_state():
    engine = VLLMRolloutEngine.__new__(VLLMRolloutEngine)
    engine._is_tp_zero = True
    engine._is_offloaded = False
    engine._partially_awake = False
    calls = []
    engine._request = lambda command, **payload: calls.append(
        (command, payload)
    )

    engine.sleep()
    assert engine._is_offloaded is True
    assert engine._partially_awake is False

    engine.wake_up(tags=["weights"])
    assert engine._is_offloaded is True
    assert engine._partially_awake is True

    engine.wake_up()
    assert engine._is_offloaded is False
    assert engine._partially_awake is False
    assert calls == [
        ("sleep", {"level": 1}),
        ("wake_up", {"tags": ["weights"]}),
        ("wake_up", {"tags": ["kv_cache"]}),
    ]
