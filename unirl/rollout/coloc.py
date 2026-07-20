"""Colocated-rollout placement + memory-dance helpers.

Two independent axes (see
``docs/2026-07-20_rollout_world_size_placement_refactor_plan.md`` §2):

- **Placement (axis A)** — how many cards ONE rollout engine actor occupies
  and how many replicas the pool fits. Verl's placement formula
  (``verl/workers/rollout/llm_server.py:417``) is the single rule:
  ``num_replicas = total_gpus // rollout_world_size``.
  This module exposes it via :func:`wire_colocated_rollout`, which
  builds one engine actor pinned to a single Worker with the engine's own
  TP group spawning inside.
- **Weight sync (axis B)** — the load and transport of the sync payload.
  Full-tensor NCCL / IPC / SGLang-tensor-bag OR LoRA adapter bag are all
  legal here; the dance below only depends on the handler's *phased
  contract* (extract/push/sync + wake/sleep), not on the concrete class.

Anchored placement is NOT specific to LoRA (that was the earlier
implementation's coupling error); full-finetune runs work by swapping the
weight-sync handler, keeping the same dance structure.

The colocate dance (rollout engine and training FSDP share the physical
cards) is:

    steady state         base ONLOADED (GPU), engines ASLEEP
    ── train_step ──
    optional sync:       onload → extract → offload
    wake engines
    optional sync:       push
    generate
    hydrate driver-side
    sleep engines
    onload base for backward

    ── evaluate ──       identical except no final onload (eval has no backward)

Callers pass ``sync_weights=True`` / ``sync_weights=False`` per step; the
scheduler handles the branch. When the engine and training FSDP live on
disjoint cards (``num_replicas`` × ``rollout_world_size == total_gpus`` but
the two pools DON'T overlap — e.g. HI3's AR:0-3 + DiT:4-7 with
train_devices=0-7 offloaded during rollout), the ``sync_weights`` gate
still governs whether ``extract/push`` fires, but the offload/onload is
NOT strictly required and can be turned off via ``offload_base``.

`unified_model.py`'s two-engine dance is the same shape (an outer loop
over engines wraps wake/sleep), and is expected to migrate onto
:class:`ColocatedRolloutScheduler` in a follow-up.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

from unirl.distributed.tensor import pytree_hydrate
from unirl.utils.hydra import parse_hydra_cfg, remote_hydra

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Placement (axis A)
# ---------------------------------------------------------------------------


def derive_placement(
    *,
    total_gpus: int,
    rollout_world_size: int,
    anchor_offset: int = 1,
) -> Tuple[int, List[int]]:
    """Verl-style placement derivation: ``(num_replicas, [anchor_device])``.

    Given the total GPUs and one engine's ``rollout_world_size`` (TP × PP × DP),
    returns the number of replicas that fit and the anchor device for each.
    Each replica's anchor is ``r * rollout_world_size + anchor_offset``
    (default ``+1`` — keeps the anchor off the train rank-0 Worker to avoid
    the RemoteLoraWeightSync.push self-deadlock; see the note in
    ``unified_model.py:260-268``). The engine's stage-yaml ``runtime.devices``
    pins its internal TP group to the physical cards ``r*rws .. r*rws+rws-1``.

    Raises when ``total_gpus`` is not divisible by ``rollout_world_size`` —
    verl's placement rule is exact, we don't silently round.
    """
    if rollout_world_size < 1:
        raise ValueError(f"rollout_world_size must be >= 1; got {rollout_world_size}")
    if total_gpus % rollout_world_size != 0:
        raise ValueError(
            f"total_gpus={total_gpus} not divisible by rollout_world_size={rollout_world_size}. "
            f"Verl placement rule requires an exact fit."
        )
    num_replicas = total_gpus // rollout_world_size
    # Anchor within each replica's slice. ``anchor_offset`` must land inside
    # the slice; clamp for the pathological ``rollout_world_size==1`` case.
    off = max(0, min(anchor_offset, rollout_world_size - 1)) if rollout_world_size > 1 else 0
    anchors = [r * rollout_world_size + off for r in range(num_replicas)]
    return num_replicas, anchors


# ---------------------------------------------------------------------------
# Engine wiring
# ---------------------------------------------------------------------------


def wire_colocated_rollout(
    *,
    pool,
    rollout_cfg,
    anchor_device: int,
) -> Any:
    """Build ONE multi-GPU rollout engine actor pinned to a single Worker.

    ``device_ids=[anchor_device]`` pins the actor to a SINGLE Worker process,
    not the whole placement scope — the engine is one TP-parallel server
    (vllm-omni's ``mp_executor`` or SGLang's SRT), not a per-device DP replica.
    Inside the engine subprocess the TP workers pin to the yaml
    ``runtime.devices`` (vLLM-Omni) or ``tp_size`` (SGLang).

    The engine takes no ``pipeline`` kwarg here (it boots its own runtime).
    Callers that want a bound-to-train-pipeline rollout (trainside engine)
    should NOT use this path — that's the SPMD ``remote()`` scatter branch.
    """
    parsed = parse_hydra_cfg(rollout_cfg)
    role_cls = parsed.pop("role_cls")
    return pool.create_remote(role_cls, device_ids=[anchor_device], init_kwargs=parsed)


def wire_colocated_weight_sync(
    *,
    sync_cfg,
    backend,
    rollout_engines: List[Any],
) -> Any:
    """Build the weight-sync handler for cross-slab pushes.

    ``sync_cfg`` MUST target a handler that supports the phased
    ``extract()`` / ``push()`` contract (currently
    :class:`RemoteLoraWeightSync`, which is what both anchored trainers use).
    We only bind ``backend=`` — the anchored engine is NOT a same-Worker
    sibling of most train ranks, so :meth:`set_rollout_targets` binds the
    rollout Workers explicitly.

    Enforces the ``verify`` capability contract: if the sync recipe asks for
    a post-load checksum read-back (``sync_cfg.verify=True``), every target
    engine class must declare ``supports_lora_checksum=True``. SGLang has no
    ``loaded_lora_checksums`` — pairing it with ``verify=True`` would fail
    silently later; we fail loud now instead.
    """
    verify = False
    try:
        verify = bool(sync_cfg.get("verify", False))
    except Exception:  # noqa: BLE001 — sync_cfg is DictConfig / dict; defensive
        verify = bool(getattr(sync_cfg, "verify", False))
    if verify:
        offenders = [
            eng.role_cls.__name__
            for eng in rollout_engines
            if not bool(getattr(getattr(eng, "role_cls", None), "supports_lora_checksum", False))
        ]
        if offenders:
            raise ValueError(
                "wire_colocated_weight_sync: sync recipe requests verify=True but "
                f"engine class(es) {offenders} do NOT declare "
                "``supports_lora_checksum=True``. Set ``sync.verify=False`` in the "
                "recipe, or switch to an engine that implements "
                "``loaded_lora_checksums`` (currently vLLM-Omni only)."
            )
    handler = remote_hydra(sync_cfg, backend=backend)
    handler.set_rollout_targets(
        [(eng.role_name, eng.workers) for eng in rollout_engines]
    )
    return handler


# ---------------------------------------------------------------------------
# Colocate memory dance (axis A — colocate variant)
# ---------------------------------------------------------------------------


class ColocatedRolloutScheduler:
    """Wrap the wake/sleep + optional offload/onload sequence around ``generate``.

    Steady state (entry to every scheduler call): base ONLOADED, engines
    ASLEEP. This matches what the outer training loop expects between steps.

    Two variants:

    - :meth:`train_step_generate`: dance + generate for training.
      Terminates with base ONLOADED (backward will read GPU params).
      Optional ``sync_weights`` gates ``extract`` / ``push``.
    - :meth:`eval_generate`: dance + generate for eval.
      Terminates with base OFFLOADED (no backward — next train_step will
      onload). Also optional ``sync_weights``.

    Both call :func:`pytree_hydrate` after ``generate`` so the anchored
    engine's driver-side ``TensorRef`` return is materialized before
    downstream ``DP_SCATTER`` dispatches try to intra-handle-slice it. On
    SPMD engines that already scatter, ``pytree_hydrate`` is a structural
    no-op.
    """

    def __init__(
        self,
        *,
        engines: List[Any],
        backend: Any,
        weight_sync: Optional[Any],
        offload_base: bool,
    ) -> None:
        self._engines = list(engines)
        self._backend = backend
        self._weight_sync = weight_sync
        # When True, the WHOLE training FSDP (frozen base + adapters + grads +
        # optimizer state) is moved to CPU while the engines are awake, and
        # back to GPU before backward. This is the ONLY correct policy when
        # the rollout engine and training FSDP share physical cards (colocate)
        # AND ``backend.cpu_offload`` is False (manual dance is the sole
        # owner of CPU↔GPU placement — the FSDP2 CPUOffloadPolicy is off).
        # False when the two pools are disjoint (train 0-7, rollout 8-15, etc.).
        self._offload_base = bool(offload_base)

    def _wake_all(self) -> None:
        for eng in self._engines:
            eng.wake_up()

    def _sleep_all(self) -> None:
        for eng in self._engines:
            eng.sleep()

    # ------------------------------------------------------------------
    # generate + memory dance (train_step & eval share the phases)
    # ------------------------------------------------------------------

    def _run_generate(self, engine: Any, req: Any) -> Any:
        """Generate + hydrate on ONE engine.

        Multi-engine (num_replicas > 1) rollouts are the caller's
        responsibility — HI3's :class:`UnifiedModelTrainer` slices the batch
        across replicas and concats the resulting tracks. This helper only
        owns the wake/sleep and generate phase for a caller that already has
        one (or many) engines lined up.
        """
        resp = engine.generate(req)
        return pytree_hydrate(resp)

    def train_step_generate(
        self,
        *,
        req: Any,
        sync_weights: bool,
        rollout_id: int = 0,
    ) -> Any:
        """One rollout dance for training.

        Entry state: base ONLOADED, engines ASLEEP.
        Exit  state: base ONLOADED (ready for backward), engines ASLEEP.
        """
        logger.debug(
            "[ROLLOUT-DANCE] train rollout %d: sync=%s wired=%s offload=%s",
            rollout_id, sync_weights, self._weight_sync is not None, self._offload_base,
        )
        if sync_weights and self._weight_sync is not None:
            # Extract with base on GPU; then offload so engines have room.
            self._weight_sync.extract()
            if self._offload_base:
                self._backend.offload()
        elif self._offload_base:
            # No sync this step — still need to vacate the cards for wake.
            self._backend.offload()

        self._wake_all()

        if sync_weights and self._weight_sync is not None:
            self._weight_sync.push()
            logger.debug("[ROLLOUT-DANCE] rollout %d: extract+push returned", rollout_id)

        # Only single-engine anchored is supported by this scheduler today;
        # HI3's two-engine slab still runs its own loop and is scheduled to
        # migrate onto this helper (see Stage 5 of the plan).
        if len(self._engines) != 1:
            raise NotImplementedError(
                f"ColocatedRolloutScheduler.train_step_generate: multi-engine "
                f"(N={len(self._engines)}) rollout is not supported yet — a "
                f"caller-side scatter/concat is required. HI3 currently handles "
                f"this in ``unified_model.run_rollout`` and will migrate here."
            )
        resp = self._run_generate(self._engines[0], req)

        self._sleep_all()
        if self._offload_base:
            self._backend.onload()
        return resp

    def eval_generate(
        self,
        *,
        req: Any,
        sync_weights: bool = False,
    ) -> Any:
        """One rollout for eval, no backward.

        Entry state: base ONLOADED, engines ASLEEP.
        Exit  state: base OFFLOADED (next train_step onloads at its end).
        """
        if sync_weights and self._weight_sync is not None:
            self._weight_sync.extract()
            if self._offload_base:
                self._backend.offload()
        elif self._offload_base:
            self._backend.offload()

        self._wake_all()
        if sync_weights and self._weight_sync is not None:
            self._weight_sync.push()

        if len(self._engines) != 1:
            raise NotImplementedError(
                "ColocatedRolloutScheduler.eval_generate: multi-engine not yet supported."
            )
        try:
            resp = self._run_generate(self._engines[0], req)
        finally:
            self._sleep_all()
            # NB: do NOT onload here — the caller (evaluate()'s outer loop
            # iterates several batches under one wake/sleep pair in the old
            # code, but we do wake/sleep per BATCH now to keep the dance
            # simple. The eval caller loops over ``eval_generate`` and
            # decides whether the final base-onload matters (it does when the
            # next call is ``train_step``, which asks for base-onloaded on
            # entry — so evaluate() must onload once at its exit).
        return resp

    # ------------------------------------------------------------------
    # Eval flow — wake ONCE across many batches (~20× cheaper than dancing
    # per batch when the eval file is large).
    # ------------------------------------------------------------------

    def enter_eval(self, *, sync_weights: bool) -> None:
        """Set up for a batch of eval calls.

        Entry state: base ONLOADED, engines ASLEEP.
        Exit  state: base OFFLOADED (colocate) / ONLOADED (SPMD sibling),
                     engines AWAKE. Follow with N :meth:`generate_batch`
                     calls, then :meth:`exit_eval` to restore steady state.
        """
        if sync_weights and self._weight_sync is not None:
            self._weight_sync.extract()
            if self._offload_base:
                self._backend.offload()
        elif self._offload_base:
            self._backend.offload()
        self._wake_all()
        if sync_weights and self._weight_sync is not None:
            self._weight_sync.push()

    def generate_batch(self, req: Any) -> Any:
        """One generate + hydrate call. Assumes engines already awake."""
        if len(self._engines) != 1:
            raise NotImplementedError(
                f"ColocatedRolloutScheduler.generate_batch: multi-engine "
                f"(N={len(self._engines)}) rollout is not supported yet."
            )
        return self._run_generate(self._engines[0], req)

    def exit_eval(self) -> None:
        """Tear down after eval batches: sleep engines, restore base to GPU."""
        self._sleep_all()
        if self._offload_base:
            self._backend.onload()


__all__ = [
    "ColocatedRolloutScheduler",
    "derive_placement",
    "wire_colocated_rollout",
    "wire_colocated_weight_sync",
]
