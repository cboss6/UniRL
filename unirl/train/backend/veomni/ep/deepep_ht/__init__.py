"""Optional DeepEP high-throughput EP transport for the VeOmni backend."""

from unirl.train.backend.veomni.ep.deepep_ht.config import DeepEPHTConfig
from unirl.train.backend.veomni.ep.deepep_ht.dispatcher import close_deepep_ht_buffers
from unirl.train.backend.veomni.ep.deepep_ht.patch import (
    install_deepep_ht_patch,
    is_deepep_ht_patch_installed,
)

__all__ = [
    "DeepEPHTConfig",
    "close_deepep_ht_buffers",
    "install_deepep_ht_patch",
    "is_deepep_ht_patch_installed",
]
