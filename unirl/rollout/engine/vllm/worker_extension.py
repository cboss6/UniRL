"""vLLM worker extension for UniRL full-weight updates."""

from __future__ import annotations

from typing import List, Optional


class UniRLWeightSyncExtension:
    """Deserialize a UniRL tensor bucket inside each TP worker."""

    def unirl_before_sleep(self) -> None:
        """Drop cached UniMatch buffers whose CuMem mappings will be released."""
        import torch

        torch.cuda.synchronize()
        from unimatch.adaptor.vllm.patches.qwen3_dualcol import qwen3_modules
        from unimatch.adaptor.vllm.patches.qwen3_moe_dualcol import (
            gate_fp32,
            grouped_runner,
        )

        qwen3_modules.close_all()
        grouped_runner.close_all()
        gate_fp32.close_all()
        torch.cuda.synchronize()

    def unirl_weight_digest(self) -> dict:
        """Return a deterministic sampled digest for cross-replica checks."""
        import hashlib

        import torch

        model = self.model_runner.get_model()
        digest = hashlib.sha256()
        count = 0
        for name, parameter in sorted(model.named_parameters()):
            flat = parameter.detach().reshape(-1)
            if not flat.numel():
                continue
            indices = torch.tensor(
                sorted({0, int(flat.numel()) // 2, int(flat.numel()) - 1}),
                dtype=torch.int64,
                device=flat.device,
            )
            sample = flat.index_select(0, indices).contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(tuple(parameter.shape)).encode("ascii"))
            digest.update(str(parameter.dtype).encode("ascii"))
            digest.update(
                sample.view(torch.uint8).cpu().numpy().tobytes()
            )
            count += 1
        return {
            "rank": int(getattr(self, "rank", 0)),
            "parameters": count,
            "sha256": digest.hexdigest(),
        }

    def unirl_weight_debug(self) -> dict:
        model = self.model_runner.get_model()
        selected = {}
        for name, parameter in model.named_parameters():
            if name.endswith(
                (
                    "embed_tokens.weight",
                    "lm_head.weight",
                    "w13_weight",
                    "w2_weight",
                )
            ):
                selected[name] = {
                    "shape": tuple(parameter.shape),
                    "values": parameter.reshape(-1)[:8].float().cpu().tolist(),
                }
        return {
            "rank": int(getattr(self, "rank", 0)),
            "parameters": selected,
        }

    def unirl_begin_debug(self, record_id: str = "batch") -> None:
        import os

        if os.environ.get("UNIMATCH_STAGE_DUMP_DIR"):
            os.environ["UNIMATCH_STAGE_DUMP_RECORD"] = str(record_id)
            from unimatch.diagnostics.stage_dump import set_stage_dump_record

            set_stage_dump_record(str(record_id))
            print(
                f"[unirl_begin_debug] stage dump record={record_id}",
                flush=True,
            )
        from unimatch.adaptor.vllm.patches.qwen3_dualcol import qwen3_modules
        from unimatch.adaptor.vllm.patches.qwen3_moe_dualcol import (
            gate_fp32,
            hf_router,
        )

        qwen3_modules._ATTENTION_DEBUGGED = False
        qwen3_modules._OPROJ_DEBUGGED = False
        qwen3_modules._LM_DEBUGGED = False
        gate_fp32._GATE_DEBUGGED = False
        hf_router._ROUTER_DEBUGGED = False

    def unirl_update_weights_from_tensor(
        self,
        *,
        serialized_named_tensors: List[str],
        load_format: Optional[str] = None,
    ) -> dict:
        if load_format not in (None, "flattened_bucket"):
            raise ValueError(f"unsupported direct-vLLM weight load format {load_format!r}")
        if not serialized_named_tensors:
            raise ValueError("direct-vLLM weight update received no payloads")

        from unirl.distributed.weight_sync.transfer.sgl_compat import (
            FlattenedTensorBucket,
            MultiprocessingSerializer,
            monkey_patch_torch_reductions,
        )

        monkey_patch_torch_reductions()
        rank = int(getattr(self, "rank", 0))
        payload = serialized_named_tensors[rank % len(serialized_named_tensors)]
        decoded = MultiprocessingSerializer.deserialize(payload)
        bucket = FlattenedTensorBucket(
            flattened_tensor=decoded["flattened_tensor"],
            metadata=decoded["metadata"],
        )
        named_tensors = bucket.reconstruct_tensors()
        model = self.model_runner.get_model()
        loaded = model.load_weights(iter(named_tensors))
        import torch

        torch.cuda.synchronize()
        return {
            "rank": rank,
            "num_tensors": len(named_tensors),
            "num_loaded": len(loaded) if loaded is not None else None,
        }


__all__ = ["UniRLWeightSyncExtension"]
