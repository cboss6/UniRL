import inspect
import logging
import time
from typing import Dict, Optional, Tuple

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from unirl.distributed.group.placement import placement, remote
from unirl.distributed.tensor import hydrate, pytree_hydrate
from unirl.rollout.coloc import (
    ColocatedRolloutScheduler,
    derive_placement,
    wire_colocated_rollout,
    wire_colocated_weight_sync,
)
from unirl.train.stack import TrainStepResult
from unirl.trainer.base import BaseTrainer, build_sampling_dict
from unirl.types.prompts import RolloutInputs
from unirl.types.rollout_req import RolloutReq
from unirl.types.sampling import BaseSamplingParams, total_samples_per_prompt
from unirl.utils.hydra import parse_hydra_cfg, remote_hydra

logger = logging.getLogger(__name__)


class ARTrainer(BaseTrainer):
    """Autoregressive (VLM / LLM) RL trainer: rollout + train colocated.

    Sibling of :class:`~unirl.trainer.diffusion.DiffusionTrainer` for the
    AR path. Structurally identical except ``_build_req`` carries **no SDE step
    scheduling** — that is diffusion-only (``DiffusionSamplingParams`` owns
    ``scheduler`` / ``sde_indices`` / ``resolve_sde_indices``), and
    ``ARSamplingParams`` has none of it. Keeping the AR trainer separate means
    the AR path never touches diffusion code (no ``hasattr`` guard, no
    ``dataclasses.replace`` of SDE fields).

    Trainside colocate (the qwen_vl recipe): the training pipeline IS the
    sampler, so ``sync_cfg`` is absent and ``weight_sync`` stays ``None``.
    """

    def __init__(
        self,
        *,
        cfg: DictConfig,
        batch_size: int,
        bundle_cfg: DictConfig,
        pipeline_cfg: DictConfig,
        backend_cfg: DictConfig,
        rollout_cfg: DictConfig,
        reward_cfg: DictConfig,
        algorithm_cfg: DictConfig,
        stack_cfg: DictConfig,
        data_source_cfg: DictConfig,
        sampling_cfg: DictConfig,
        sync_cfg: Optional[DictConfig] = None,
        logging_cfg: Optional[DictConfig] = None,
        adv_normalization_scope: str = "group",
        normalize_adv_by_std: bool = True,
        balance_shards: bool = False,
        eval_interval: int = 0,
        eval_num_prompts: int = -1,
        eval_batch_size: int = 8,
        eval_samples_per_prompt: int = 16,
        eval_temperature: float = 1.0,
        # ---- Legacy placement overrides (DEPRECATED) ----
        # Both derive from the engine config's ``rollout_world_size`` when
        # left as ``None`` (verl formula, see ``unirl/rollout/coloc.py``).
        # Kept as opt-in overrides for the (rare) case where a recipe wants
        # to force a non-standard anchor or explicitly turn base-offload
        # off/on; new recipes should NOT set them.
        rollout_anchor_device: Optional[int] = None,
        enable_fsdp_offload: Optional[bool] = None,
    ) -> None:
        super().__init__(cfg=cfg, logging_cfg=logging_cfg)
        self.batch_size = batch_size
        # "group" (textbook GRPO, default) or "global" (v1 baseline parity).
        self.adv_normalization_scope = adv_normalization_scope
        # True (default) = standard GRPO: divide the group-relative advantage by the
        # group std. False = mean-center only (reward - group_mean), NO std division —
        # removes the difficulty bias that over-amplifies low-std (hard) prompts.
        self.normalize_adv_by_std = normalize_adv_by_std
        # verl trainer.balance_batch parity: driver-side reorder of the rollout
        # batch so each DP shard receives a similar total-token workload. FSDP
        # collectives sync all ranks every micro, so a step runs at the SLOWEST
        # rank's pace — without balancing, the rank that drew the longest
        # sequences straggles (~+/-11%% rank-total variance at heavy lengths).
        self.balance_shards = bool(balance_shards)  # overrides the BaseTrainer default (False)
        # AIME-style periodic eval — avg@k accuracy on the eval prompt set
        # (run.eval_data_path), logged under eval/*. eval_interval=0 disables it.
        # ``eval_num_prompts`` sentinel:
        #   -1 (default, or any negative)  → full eval set
        #    0                             → yield nothing (explicit skip)
        #    N > 0                         → cap: score first N prompts
        # ``eval_batch_size`` (default 8) is the iteration batch size, decoupled
        # from the eval-set size (mirrors verl's ``data.val_batch_size``). Bounds
        # peak GPU memory during eval-time rollout.
        self.eval_interval = int(eval_interval)
        _num = int(eval_num_prompts)
        self.eval_num_prompts = -1 if _num < 0 else _num
        self.eval_batch_size = max(1, int(eval_batch_size))
        self.eval_samples_per_prompt = int(eval_samples_per_prompt)
        self.eval_temperature = float(eval_temperature)

        # ------------------------------------------------------------------
        # Placement derivation (verl formula, see unirl/rollout/coloc.py).
        # ------------------------------------------------------------------
        # Read the engine's world_size off the config WITHOUT booting the
        # engine (Hydra ``_recursive_=False`` skips deeper instantiations).
        # An engine that does not expose ``rollout_world_size`` falls back
        # to the base class default (1 = SPMD replica-per-rank).
        engine_config_local = None
        if rollout_cfg is not None and "config" in rollout_cfg:
            try:
                engine_config_local = instantiate(rollout_cfg.config, _recursive_=False)
            except Exception as exc:  # noqa: BLE001 — best-effort, degrade to SPMD
                logger.debug("could not pre-instantiate rollout config: %s", exc)
        rws = int(getattr(engine_config_local, "rollout_world_size", 1) or 1)

        # Multi-actor placement is derived; a legacy ``rollout_anchor_device``
        # override collapses to a single-anchor placement for backwards compat.
        # ``num_replicas`` > 1 is not yet supported by ARTrainer's train_step
        # (needs a driver-side scatter/concat like HI3's run_rollout — a
        # follow-up); we assert === 1 below.
        if rollout_anchor_device is not None:
            num_replicas = 1
            anchors = [int(rollout_anchor_device)]
            self._is_anchored = True
        elif rws > 1:
            num_replicas, anchors = derive_placement(
                total_gpus=self.pool.num_devices,
                rollout_world_size=rws,
            )
            self._is_anchored = True
        else:
            num_replicas, anchors = 1, []
            self._is_anchored = False

        if self._is_anchored and num_replicas != 1:
            raise NotImplementedError(
                f"ARTrainer: rollout_world_size={rws} on {self.pool.num_devices} GPUs "
                f"would place {num_replicas} anchored replicas, but the current "
                f"train_step / evaluate path only supports 1. Use a stage yaml whose "
                f"tensor_parallel_size == total_gpus, or wait for the multi-replica "
                f"scatter/concat migration."
            )

        # ------------------------------------------------------------------
        # Base-offload policy — colocate implies True by default.
        # ------------------------------------------------------------------
        if enable_fsdp_offload is None:
            # Colocate (anchored engine shares the physical cards with training
            # FSDP) requires manual CPU↔GPU. SPMD replica-per-rank stays on GPU
            # (the engine shares the module with training, no separate placement).
            enable_fsdp_offload = self._is_anchored
        self._enable_fsdp_offload = bool(enable_fsdp_offload)

        # Legacy attribute kept for :meth:`train_step` / :meth:`evaluate` — they
        # still branch on ``_rollout_anchor_device is None`` to pick SPMD vs
        # anchored. Assigning the derived anchor here preserves that branch
        # without exposing anchor selection to users.
        self._rollout_anchor_device: Optional[int] = anchors[0] if self._is_anchored else None

        # ------------------------------------------------------------------
        # Front-of-init assertions on placement + memory ownership.
        # ------------------------------------------------------------------
        # Colocate manual dance requires ``backend.cpu_offload == False`` (a
        # second placement owner — FSDP2's CPUOffloadPolicy — fires
        # ``_validate_cpu_offload_params`` the moment we onload). This is a
        # policy assertion on the recipe: it doesn't depend on the engine
        # or the weight-sync handler, only on the "colocate + auto offload".
        if self._is_anchored and self._enable_fsdp_offload:
            fsdp_cfg = backend_cfg.get("fsdp_cfg") if isinstance(backend_cfg, DictConfig) else None
            if fsdp_cfg is not None:
                cpu_off = fsdp_cfg.get("cpu_offload", False)
                if bool(cpu_off):
                    raise ValueError(
                        "ARTrainer: anchored + enable_fsdp_offload requires "
                        "backend.fsdp_cfg.cpu_offload=False. The manual dance owns "
                        "CPU↔GPU placement exclusively; FSDP2's CPUOffloadPolicy "
                        "as a second owner fires _validate_cpu_offload_params on "
                        "onload. See unirl/rollout/coloc.py for the ownership rule."
                    )

        # Driver-side data iterator (not a Remote).
        self.data_source = instantiate(data_source_cfg)

        self.sampling_params: Dict[str, BaseSamplingParams] = build_sampling_dict(sampling_cfg)

        # Set below from the `sync` block; None trainside (shares the module).
        self.weight_sync = None
        # Colocate scheduler owns the wake/sleep + offload/onload sequence when
        # ``_is_anchored``. SPMD path stays inline in train_step / evaluate
        # (no dance — engine is a train-rank sibling, memory sharing is direct).
        self._rollout_scheduler: Optional[ColocatedRolloutScheduler] = None

        with placement(self.pool, fraction=1.0, shared_workers=True):
            self.bundle = remote_hydra(bundle_cfg)
            self.pipeline = remote_hydra(pipeline_cfg, bundle=self.bundle)
            self.backend = remote_hydra(backend_cfg, bundle=self.bundle)

            self.reward = remote_hydra(reward_cfg)
            self.algorithm = remote_hydra(algorithm_cfg, pipeline=self.pipeline)
            self.stack = remote_hydra(stack_cfg, fsdp_backend=self.backend, algorithm=self.algorithm)

            if not self._is_anchored:
                # SPMD path (default): rollout scattered to every rank's slot0,
                # constructed alongside the other roles.
                rollout_parsed = parse_hydra_cfg(rollout_cfg)
                if "pipeline" in inspect.signature(rollout_parsed["role_cls"]).parameters:
                    self.rollout = remote(**rollout_parsed, pipeline=self.pipeline)  # for direct sampling
                else:
                    self.rollout = remote(**rollout_parsed)  # for vllm / sglang TP=1
                if sync_cfg is not None:
                    # SPMD sibling engine → LocalLoraWeightSync (in-process copy).
                    self.weight_sync = remote_hydra(sync_cfg, backend=self.backend, rollout=self.rollout)
            else:
                # Anchored (colocate) path: rollout engine is one multi-GPU actor
                # pinned to a single Worker. Boot procedure (matches the
                # ColocatedRolloutScheduler entry contract):
                #   1. Offload the FSDP base to CPU (frees the physical cards
                #      for vllm-omni's TP workers → passes gpu_memory_utilization).
                #   2. Wire the anchored engine and sleep it (steady state).
                #   3. Wire the weight-sync handler (backend-only; targets set).
                #   4. Build the scheduler that owns future wake/sleep + offload
                #      sequencing.
                if self._enable_fsdp_offload:
                    self.backend.offload()

                self.rollout = wire_colocated_rollout(
                    pool=self.pool,
                    rollout_cfg=rollout_cfg,
                    anchor_device=self._rollout_anchor_device,
                )
                self.rollout.sleep()

                if sync_cfg is not None:
                    self.weight_sync = wire_colocated_weight_sync(
                        sync_cfg=sync_cfg,
                        backend=self.backend,
                        rollout_engines=[self.rollout],
                    )

                self._rollout_scheduler = ColocatedRolloutScheduler(
                    engines=[self.rollout],
                    backend=self.backend,
                    weight_sync=self.weight_sync,
                    offload_base=self._enable_fsdp_offload,
                )

    def _build_req(self, inputs: RolloutInputs, rollout_id: int) -> RolloutReq:
        """Turn a data source batch into a typed :class:`RolloutReq`.

        Expands ``inputs`` by ``total_samples_per_prompt(sampling_params)`` so
        each prompt produces an N-sample GRPO group (sibling samples consecutive).
        AR sampling params ride to the engine untouched — there is no SDE step
        schedule to resolve (that is the diffusion trainer's job).
        """
        inputs = inputs.expand(total_samples_per_prompt(self.sampling_params))
        req = RolloutReq(
            sample_ids=list(inputs.sample_ids),
            group_ids=list(inputs.group_ids),
            primitives=dict(inputs.primitives),
            request_conditions={},
            sampling_params=self.sampling_params,
            metadata=list(inputs.metadata) if inputs.metadata else [],
        )
        return req

    def train_step(
        self,
        req: RolloutReq,
        *,
        training_progress: float = 0.0,
        sync_weights: bool = False,
        rollout_id: int = 0,
    ) -> Tuple[TrainStepResult, float]:
        """One ``rollout → reward → advantage → optimizer step`` pass.

        Returns ``(train_result, mean_reward)`` — the mean unnormalized
        per-sample reward of the single track (0.0 if none), for the log line.
        ``rollout_id`` only keys the wandb panels (see :meth:`UniRLWandBLogger.log_rollout_step`).
        """
        t0 = time.perf_counter()
        if not self._is_anchored:
            # SPMD path: rollout sibling of every train rank; sync + generate + sleep in-place.
            self.rollout.wake_up()
            if sync_weights and self.weight_sync is not None:
                self.weight_sync.sync()
            resp = self.rollout.generate(req)
            self.rollout.sleep()
        else:
            # Colocate memory dance owned by the scheduler:
            #   entry: base ONLOADED, engine ASLEEP
            #   → (extract → offload → wake → push → generate → hydrate → sleep → onload)
            #   exit:  base ONLOADED (ready for backward), engine ASLEEP
            # Full sequence and rationale in ``unirl/rollout/coloc.py``.
            resp = self._rollout_scheduler.train_step_generate(
                req=req, sync_weights=sync_weights, rollout_id=rollout_id,
            )

        for name, track in list(resp.tracks.items()):
            if track.segment is not None:
                resp.tracks[name] = self.reward.score_and_attach(req=req, track=track)

        mean_reward = 0.0
        for track in resp.tracks.values():
            if track.rewards is None:
                continue
            # Hydrate in place so the wandb reward/advantage stats reuse this
            # fetch instead of re-pulling the TensorRef from the worker.
            track.rewards = hydrate(track.rewards)
            mean_reward = float(track.rewards.to(torch.float32).mean().item())
            break  # single-track for now; revisit if multi-track lands

        for name, track in list(resp.tracks.items()):
            if track.rewards is not None:
                resp.tracks[name] = track.compute_advantages(
                    normalize=self.normalize_adv_by_std, scope=self.adv_normalization_scope
                )

        self._dump_rollout_samples(req, resp, rollout_id)
        self._drop_decoded(req, resp, rollout_id=rollout_id)
        (track,) = resp.tracks.values()
        # verl balance_batch parity: reorder so each DP shard gets a near-equal
        # token load before DP_SCATTER (no-op when already balanced).
        if self.balance_shards:
            track = track.balance_shards(int(self.num_devices))
        result = self.stack.train_track(track, training_progress=float(training_progress))
        self.wandb_logger.log_rollout_step(
            rollout_id,
            result,
            resp,
            step_time_s=time.perf_counter() - t0,
            trunc_len=getattr(self.sampling_params.get("ar"), "max_new_tokens", None),
        )
        return result, mean_reward

    def evaluate(self, rollout_id: int) -> float:
        """Periodic eval — ``avg@k`` accuracy on the eval prompt set.

        Mirrors :meth:`train_step`'s rollout+reward path but skips
        advantage/backward: iterate up to ``eval_num_prompts`` prompts from
        ``run.eval_data_path`` in ``eval_batch_size``-sized batches, expand
        each prompt to ``eval_samples_per_prompt`` siblings, generate at
        ``eval_temperature``, score, and log the mean reward under both
        ``eval/acc`` (= avg@k accuracy, since the MC reward is 0/1) and
        ``eval/reward`` (shares the eval axis with the other trainers).
        Returns it.

        ``eval_num_prompts=-1`` (default) evaluates the full eval set;
        ``eval_num_prompts=0`` yields no batches (explicit skip). See the
        sentinel table on :meth:`~unirl.data.data_source.MultimodalRLDataSource.iter_eval_batches`.
        """
        import dataclasses

        eval_ar = dataclasses.replace(
            self.sampling_params.get("ar"),
            samples_per_prompt=self.eval_samples_per_prompt,
            temperature=self.eval_temperature,
        )
        eval_sp = {**self.sampling_params, "ar": eval_ar}
        eval_batches = self.data_source.iter_eval_batches(
            self.eval_batch_size,
            eval_num_prompts=self.eval_num_prompts,
        )
        reward_sum, reward_n, prompt_n, batch_n = 0.0, 0, 0, 0

        anchored = self._is_anchored

        # Enter eval: (colocate) extract → offload → wake → push. SPMD: just
        # wake + (optional) sync. Same phased contract the scheduler uses in
        # train_step, but we hold wake ACROSS all eval batches (~20× cheaper
        # than dancing per-batch) so we open/close explicitly here.
        # The manual dance moves the WHOLE model (cpu_offload=False, sole owner
        # of CPU↔GPU placement) — vacates everything so the rollout engine fits.
        if anchored:
            self._rollout_scheduler.enter_eval(
                sync_weights=self.weight_sync is not None,
            )
        else:
            self.rollout.wake_up()
            if self.weight_sync is not None:
                self.weight_sync.sync()
        try:
            for eval_inputs in eval_batches:
                batch_n += 1
                prompt_n += len(eval_inputs.sample_ids)
                inputs = eval_inputs.expand(self.eval_samples_per_prompt)
                req = RolloutReq(
                    sample_ids=list(inputs.sample_ids),
                    group_ids=list(inputs.group_ids),
                    primitives=dict(inputs.primitives),
                    request_conditions={},
                    sampling_params=eval_sp,
                    metadata=list(inputs.metadata) if inputs.metadata else [],
                )
                if anchored:
                    # Scheduler generate + hydrate (structural no-op on SPMD).
                    resp = self._rollout_scheduler.generate_batch(req)
                else:
                    resp = self.rollout.generate(req)
                for track in resp.tracks.values():
                    if track.segment is not None:
                        track = self.reward.score_and_attach(req=req, track=track)
                    if track.rewards is not None:
                        rewards = hydrate(track.rewards).to(torch.float32)
                        reward_sum += float(rewards.sum().item())
                        reward_n += int(rewards.numel())
                        break  # single-track for now; revisit if multi-track lands
        finally:
            if anchored:
                self._rollout_scheduler.exit_eval()
            else:
                self.rollout.sleep()

        acc = reward_sum / max(1, reward_n)
        logger.info(
            "EVAL rollout %d  eval_acc(avg@%d over %d prompts, %d batches of <=%d)=%.4f",
            rollout_id + 1,
            self.eval_samples_per_prompt,
            prompt_n,
            batch_n,
            self.eval_batch_size,
            acc,
        )
        # MC reward is 0/1 so mean reward == accuracy; also emit it as `reward`
        # so this run shares the eval/reward axis with the other trainers.
        self.wandb_logger.log_eval(rollout_id + 1, {"acc": acc, "reward": acc})
        return acc

    def _dump_rollout_samples(self, req, resp, rollout_id: int) -> None:
        """Debug dump of the first N (prompt, output, reward) triples per rollout.

        Off unless ``ROLLOUT_DUMP_DIR`` is set (driver-side env). Writes one
        ``rollout_<id>.jsonl`` per rollout (``ROLLOUT_DUMP_N`` samples, default
        4) so rollout-engine quality can be eyeballed without keeping the full
        decoded batch alive. Must run BEFORE ``_drop_decoded``. Never raises.
        (Ported from the b182a511 LIN-371 lineage — lost in the rebase.)
        """
        import json
        import os

        out_dir = os.environ.get("ROLLOUT_DUMP_DIR", "")
        if not out_dir:
            return
        try:
            n = int(os.environ.get("ROLLOUT_DUMP_N", "4"))
            prompts = getattr(req.primitives.get("text"), "texts", None) or []
            (track,) = resp.tracks.values()
            outputs = getattr(track.decoded, "texts", None) or []
            rewards = track.rewards.to(torch.float32).tolist() if track.rewards is not None else []
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, f"rollout_{int(rollout_id):04d}.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                for i in range(min(n, len(outputs))):
                    f.write(
                        json.dumps(
                            {
                                "rollout": int(rollout_id),
                                "sample": i,
                                "prompt": prompts[i] if i < len(prompts) else None,
                                "output": outputs[i],
                                "output_chars": len(outputs[i] or ""),
                                "reward": rewards[i] if i < len(rewards) else None,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
        except Exception as exc:  # debug path — never let it kill training
            logger.warning("rollout sample dump failed: %s", exc)

    def train(
        self,
        *,
        num_rollouts: int,
        weight_sync_interval: int = 1,
        save_interval: int = 0,
        save_dir: Optional[str] = None,
        load_dir: Optional[str] = None,
        save_mode: str = "auto",
    ) -> None:
        """Minimal training loop: ``num_rollouts`` iterations of ``train_step``.

        ``weight_sync_interval``: sync the adapter into the engine every N
        rollouts (fused into ``train_step``'s generate; no-op trainside).

        ``save_interval``: write a checkpoint every N rollouts (and on the last
        one); ``0`` disables it. ``save_dir`` is the output folder (defaults to
        ``./checkpoints``); ``save_mode="auto"`` writes LoRA-only checkpoints
        when LoRA is active and full checkpoints otherwise.
        ``load_dir``: restore from a checkpoint directory and RESUME from its
        saved step — ``num_rollouts`` is the TOTAL budget.

        Deferred: ``num_updates_per_batch`` multi-epoch replay, eval cadence.
        """
        interval = max(1, weight_sync_interval)
        start_rollout = self.maybe_load_checkpoint(load_dir, num_rollouts=num_rollouts)
        resumed = bool(load_dir)
        # Fast-forward the data stream to the resume point — exact when
        # run.seed is set (deterministic shuffle); with seed=null the stream
        # is non-reproducible anyway.
        for _ in range(start_rollout):
            self.data_source.get_samples(self.batch_size)
        self._init_wandb(
            num_rollouts=num_rollouts,
            extra={"adv_normalization_scope": self.adv_normalization_scope},
        )
        try:
            if self.eval_interval > 0:
                self.evaluate(rollout_id=-1)  # baseline AIME accuracy, logged at eval step 0
            for rollout_id in range(start_rollout, num_rollouts):
                training_progress = rollout_id / max(1, num_rollouts - 1)
                inputs = self.data_source.get_samples(self.batch_size)
                req = self._build_req(inputs, rollout_id)
                # Sync before generate; skip step 0 (nothing trained yet). On
                # resume, force the first sync — the engine booted with fresh
                # weights and needs the restored adapter before generate.
                sync_weights = (rollout_id > 0 and rollout_id % interval == 0) or (
                    resumed and rollout_id == start_rollout
                )
                result, mean_reward = self.train_step(
                    req,
                    training_progress=training_progress,
                    sync_weights=sync_weights,
                    rollout_id=rollout_id,
                )
                self.wandb_logger.log_progress(rollout_id, num_rollouts, result, mean_reward, logger=logger)
                if self.eval_interval > 0 and (rollout_id + 1) % self.eval_interval == 0:
                    self.evaluate(rollout_id=rollout_id)
                self.maybe_save_checkpoint(
                    rollout_id, num_rollouts, save_interval=save_interval, save_dir=save_dir, save_mode=save_mode
                )
        finally:
            self._finish_wandb()
