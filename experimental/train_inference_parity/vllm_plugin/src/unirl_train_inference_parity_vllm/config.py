"""Environment-only configuration available in every spawned vLLM process."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class PluginConfig:
    profile: str
    model: str
    patches: tuple[str, ...]
    strict: bool


def load_config() -> PluginConfig:
    profile = os.environ.get("UNIRL_PARITY_PROFILE", "public_reference").strip().lower()
    if profile not in {"public_reference", "optimized"}:
        raise ValueError(f"unknown UNIRL_PARITY_PROFILE={profile!r}")
    if (
        profile == "optimized"
        and os.environ.get(
            "UNIRL_PARITY_OPTIMIZED_LICENSE_ACCEPTED",
            "0",
        )
        != "1"
    ):
        raise RuntimeError(
            "optimized parity profile is unavailable until kernel provenance and license review is complete"
        )
    model = (
        os.environ.get(
            "UNIRL_PARITY_MODEL",
            "qwen3_moe_30b_a3b",
        )
        .strip()
        .lower()
    )
    raw_patches = os.environ.get(
        "UNIRL_PARITY_PATCHES",
        "common,qwen3_moe_30b_a3b",
    )
    patches = tuple(value.strip() for value in raw_patches.split(",") if value.strip())
    strict = os.environ.get("UNIRL_PARITY_STRICT", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    return PluginConfig(profile=profile, model=model, patches=patches, strict=strict)


__all__ = ["PluginConfig", "load_config"]
