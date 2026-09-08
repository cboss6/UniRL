from __future__ import annotations

import importlib.metadata

import pytest

from experimental.train_inference_parity.lifecycle import PLUGIN_ENTRYPOINT
from experimental.train_inference_parity.profiles import (
    ParityProfile,
    require_profile_available,
    resolve_profile,
)


def test_editable_plugin_entrypoint_is_installed() -> None:
    entries = importlib.metadata.entry_points().select(group="vllm.general_plugins")
    assert PLUGIN_ENTRYPOINT in {entry.name for entry in entries}


def test_public_profile_is_default() -> None:
    profile = resolve_profile(ParityProfile.PUBLIC_REFERENCE)
    require_profile_available(profile)
    assert profile.providers["grouped"] == "vllm_bi_loop"


def test_optimized_profile_fails_closed_without_license(monkeypatch) -> None:
    monkeypatch.delenv("UNIRL_PARITY_OPTIMIZED_LICENSE_ACCEPTED", raising=False)
    with pytest.raises(RuntimeError, match="provenance/license"):
        require_profile_available(resolve_profile(ParityProfile.OPTIMIZED))
