"""Worker-extension class installed on the Qwen3-Omni thinker AR stage of vllm-omni.

This is the ``engine_args.worker_extension_cls`` qualname the Qwen3-Omni stage
YAMLs (``stage_configs/qwen3_omni_thinker_only_rl*.yaml``) reference; vllm-omni
composes it onto the AR worker (``GPUARWorker``) at instantiation. Without it the
worker has no ``set_lora_from_tensor_dict[_copy]`` and the trainer's
``RemoteLoraWeightSync.push`` fails with ``'GPUARWorker' object has no attribute
'set_lora_from_tensor_dict_copy'`` — the adapter is never registered and rollout
falls back to a (nonexistent) on-disk load.

Composes the two model-agnostic receive mixins:

- ``BucketedIPCReceiveMixin`` — bucketed CUDA-IPC ``update_weights_from_ipc`` +
  the LoRA tensor-bag receivers ``set_lora_from_tensor_dict`` (zero-copy handle)
  and ``set_lora_from_tensor_dict_copy`` (byte-copy, TP>1-safe — what the recipe's
  ``sync.copy: true`` uses). Its ``__new__`` installs the generic
  ``VLLMOmniHijack`` so ``add_lora`` accepts tensor-bag requests.
- ``NcclBroadcastReceiveMixin`` — SGLang-shape NCCL primitives, for base-weight
  broadcast sync (unused by the LoRA-only path but kept for parity with the
  other stage extensions).

Deliberately does NOT inherit ``HI3ARWeightSyncExtension``: that pulls in
``HI3ARWorkerExtension`` (``patches/compat_tokenizer``), whose import installs two
HunyuanImage3-specific patches — the ``convert_tokens_to_ids`` ratio-token fixup
(for HI3's Base ckpt) and the ``get_moe_expert_mapping`` 2-tuple unwrap
(``compat_hi3_lora``). Qwen3-Omni thinker needs neither: it boots with
``enable_lora=true`` without the MoE-mapping patch (the AR stage allocates KV
cache with no ``get_moe_expert_mapping`` error), and its Instruct tokenizer has no
missing ratio tokens. Composing only the transport mixins keeps the HI3 quirks
off the Qwen3-Omni worker.

Host contract satisfied by ``GPUARWorker`` (→ upstream ``vllm.v1.worker.gpu_worker.Worker``):
``device``, ``local_rank``, ``add_lora(req)`` / ``remove_lora(int_id)``, and the
model's ``load_weights`` at ``self.model_runner.model`` (``BucketedIPCReceiveMixin``
probes both the DiT and AR layouts).
"""

from __future__ import annotations

from unirl.rollout.engine.vllm_omni.worker.ipc_receive_mixin import (
    BucketedIPCReceiveMixin,
)
from unirl.rollout.engine.vllm_omni.worker.nccl_receive_mixin import (
    NcclBroadcastReceiveMixin,
)


class Qwen3OmniARWeightSyncExtension(
    BucketedIPCReceiveMixin,
    NcclBroadcastReceiveMixin,
):
    """Receive-side extension for the Qwen3-Omni thinker AR stage."""

    pass


__all__ = ["Qwen3OmniARWeightSyncExtension"]
