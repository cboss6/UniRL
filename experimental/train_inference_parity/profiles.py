"""Named provider profiles for the parity experiment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Mapping


class ParityProfile(str, Enum):
    PUBLIC_REFERENCE = "public_reference"
    OPTIMIZED = "optimized"


@dataclass(frozen=True)
class ProfileConfig:
    name: ParityProfile
    providers: Mapping[str, str]
    requires_licensed_kernels: bool


_PUBLIC = ProfileConfig(
    name=ParityProfile.PUBLIC_REFERENCE,
    providers={
        "norm": "vllm_bi",
        "norm_backward": "torch",
        "reductions": "vllm_bi",
        "dense": "vllm_bi",
        "grouped": "vllm_bi_loop",
        "gate": "vllm_bi",
        "router_softmax": "vllm_bi",
        "moe_combine": "torch",
    },
    requires_licensed_kernels=False,
)

_OPTIMIZED = ProfileConfig(
    name=ParityProfile.OPTIMIZED,
    providers={
        "norm": "experimental",
        "norm_backward": "experimental",
        "reductions": "vllm_bi",
        "dense": "experimental",
        "grouped": "experimental",
        "gate": "experimental_fp32",
        "router_softmax": "vllm_bi",
        "moe_combine": "experimental",
    },
    requires_licensed_kernels=True,
)


def resolve_profile(value: str | ParityProfile | None = None) -> ProfileConfig:
    raw = value or os.environ.get("UNIRL_PARITY_PROFILE", ParityProfile.PUBLIC_REFERENCE.value)
    profile = raw if isinstance(raw, ParityProfile) else ParityProfile(str(raw))
    return _PUBLIC if profile is ParityProfile.PUBLIC_REFERENCE else _OPTIMIZED


def require_profile_available(profile: ProfileConfig) -> None:
    if not profile.requires_licensed_kernels:
        return
    if os.environ.get("UNIRL_PARITY_OPTIMIZED_LICENSE_ACCEPTED", "0") != "1":
        raise RuntimeError(
            "optimized parity profile is unavailable in the open-source experiment: "
            "experiment-owned kernels have not completed provenance/license review. "
            "Use UNIRL_PARITY_PROFILE=public_reference."
        )


__all__ = [
    "ParityProfile",
    "ProfileConfig",
    "require_profile_available",
    "resolve_profile",
]
