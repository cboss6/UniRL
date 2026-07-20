"""Historical shim — ``deep_hydrate`` has been renamed and relocated.

The canonical implementation now lives at
:func:`unirl.distributed.tensor.pytree.pytree_hydrate` (also re-exported via
``unirl.distributed.tensor``). This module remains only as a backwards-compat
re-export so external callers importing ``unirl.trainer._hydrate.deep_hydrate``
keep working. New code should import from ``unirl.distributed.tensor``.
"""

from __future__ import annotations

from unirl.distributed.tensor import pytree_hydrate as deep_hydrate

__all__ = ["deep_hydrate"]
