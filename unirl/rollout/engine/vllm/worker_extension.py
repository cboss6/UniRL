"""vLLM worker extension for UniRL full-weight updates."""

from __future__ import annotations

from typing import List, Optional


class UniRLWeightSyncExtension:
    """Deserialize a UniRL tensor bucket inside each TP worker."""

    def unirl_before_sleep(self) -> None:
        """Drop model caches whose CuMem mappings will be released."""
        import torch

        torch.cuda.synchronize()
        model = self.model_runner.get_model()
        for module in model.modules():
            cleanup = getattr(module, "_unirl_before_sleep", None)
            if callable(cleanup):
                cleanup()
        torch.cuda.synchronize()

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
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        tp_world_size = int(get_tensor_model_parallel_world_size())
        tp_rank = int(get_tensor_model_parallel_rank())
        if len(serialized_named_tensors) != tp_world_size:
            raise ValueError(
                f"direct-vLLM payload count {len(serialized_named_tensors)} "
                f"does not match TP world size {tp_world_size}"
            )
        if tp_rank < 0 or tp_rank >= tp_world_size:
            raise ValueError(f"direct-vLLM TP rank {tp_rank} outside world size {tp_world_size}")
        payload = serialized_named_tensors[tp_rank]
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
            "rank": tp_rank,
            "num_tensors": len(named_tensors),
            "num_loaded": len(loaded) if loaded is not None else None,
        }


__all__ = ["UniRLWeightSyncExtension"]
