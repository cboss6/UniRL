"""Runtime manifest and plugin discovery checks."""

from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path

from .profiles import ProfileConfig, require_profile_available

PLUGIN_ENTRYPOINT = "unirl_train_inference_parity"
PLUGIN_DISTRIBUTION = "unirl-train-inference-parity-vllm"


def apply_profile_environment(profile: ProfileConfig) -> None:
    require_profile_available(profile)
    os.environ["UNIRL_PARITY_PROFILE"] = profile.name.value
    os.environ["UNIRL_PARITY_MODEL"] = "qwen3_moe_30b_a3b"
    os.environ["UNIRL_PARITY_STRICT"] = "1"
    os.environ["VLLM_PLUGINS"] = PLUGIN_ENTRYPOINT
    for name, value in profile.providers.items():
        os.environ[f"UNIRL_PARITY_{name.upper()}_PROVIDER"] = value


def require_vllm_plugin_installed() -> None:
    entries = importlib.metadata.entry_points().select(group="vllm.general_plugins")
    names = {entry.name for entry in entries}
    if PLUGIN_ENTRYPOINT not in names:
        raise RuntimeError(
            f"missing vLLM plugin entry point {PLUGIN_ENTRYPOINT!r}; install it with "
            "`pip install -e experimental/train_inference_parity/vllm_plugin`"
        )


def write_launch_manifest(path: str | os.PathLike[str], *, profile: ProfileConfig) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "train_inference_parity",
        "plugin_entrypoint": PLUGIN_ENTRYPOINT,
        "profile": profile.name.value,
        "providers": dict(profile.providers),
    }
    destination.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")


__all__ = [
    "PLUGIN_DISTRIBUTION",
    "PLUGIN_ENTRYPOINT",
    "apply_profile_environment",
    "require_vllm_plugin_installed",
    "write_launch_manifest",
]
