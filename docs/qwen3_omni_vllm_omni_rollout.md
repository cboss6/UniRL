# Qwen3-Omni Thinker · vLLM-Omni Rollout

GSPO + LoRA RL training for the **Qwen3-Omni thinker** with rollout served by the
**vLLM-Omni** engine. Training runs trainside (FSDP DP=8); only rollout generation
is delegated to vLLM-Omni.

## Topology

The vLLM-Omni engine is **anchored to a single Worker actor** and runs
tensor-parallel (TP=4) across 4 GPUs. Training runs FSDP DP=8 across all 8 GPUs.
The two share the physical cards by time-slicing:

1. Before each rollout, the whole training model (frozen base + encoders + LoRA,
   plus grads / optimizer state) is **offloaded to CPU** and the caching allocator
   is emptied, freeing the GPUs for the engine's TP workers.
2. The freshly-trained LoRA adapter is pushed into the engine.
3. The engine wakes, generates, then sleeps.
4. The base is **onloaded** back to GPU for the backward pass.

`FSDP2 fully_shard` owns *sharding* only; the manual offload/onload dance is the
sole owner of CPU↔GPU placement (hence `fsdp_cfg.cpu_offload: false` in the
recipe — a second placement owner would conflict).

TP is **4, not 8**: the thinker's audio tower has 20 attention heads (loaded even
with `use_audio_in_video: false`), and TP must divide 20.

## Key files

| File | Role |
| --- | --- |
| `unirl/models/qwen3_omni/` | Trainside model bundle: `Qwen3OmniBundle`, `Qwen3OmniPipeline`, the AR stage (`autoregress` + teacher-forced `replay`), chat-template stage, and typed conditions. |
| `unirl/rollout/engine/vllm_omni/adapters/qwen3_omni.py` | Rollout adapter (`modality: qwen3_omni_thinker`): driver-side processor encode → vLLM-Omni AR prompt; replay-condition assembly. |
| `unirl/rollout/engine/vllm_omni/patches/qwen3_omni.py` | Boot patches: register the thinker with `AutoModelForCausalLM`, FSDP-init fixups, processor/tokenizer extensions. |
| `unirl/rollout/engine/vllm_omni/worker/qwen3_omni_ar_extension.py` | Worker-side weight-sync receive extension (`set_lora_from_tensor_dict[_copy]`). |
| `unirl/rollout/engine/vllm_omni/stage_configs/qwen3_omni_thinker_only_rl*.yaml` | vLLM-Omni stage configs (TP=8 default, TP=4 twin). |
| `unirl/trainer/ar.py` | `ARTrainer` — the anchored-rollout topology (`rollout_anchor_device`, the colocate offload/onload dance). |
| `examples/ar/qwen3_omni_video_r1_gspo_lora_vllm_omni_1x8.yaml` | Recipe. |

## LoRA weight sync

`RemoteLoraWeightSync` (rank-0 raw-ray push) with:

- `param_prefix: "thinker."` — the thinker loads standalone (HF strips the
  `thinker.` prefix), but vLLM-Omni's LoRA key mapper expects `thinker.model.*`.
  Re-adding the prefix makes the mapper resolve the real modules; without it the
  engine loads 0 adapters and silently runs base weights.
- `copy: true` — the byte-copy transport, required for TP>1 (zero-copy IPC handles
  crash ranks 2..N).
- `verify: true` — a packing-aware checksum read-back that catches a silent
  0-adapter load each sync.

## Run

```bash
bash run_vllm_omni.sh
# quick check (few rollouts, no eval / wandb):
bash run_vllm_omni.sh num_rollouts=3 eval_interval=0 logging.report_to_wandb=false
```

Environment: `QWEN3_OMNI_PATH` (model dir), `DATA_PATH` / `EVAL_DATA_PATH`
(jsonl). Build the dataset with
`scripts/convert_video_r1_260k_to_unirl.py`.

## Data

Video multiple-choice (Video-R1-260k). Each prompt carries a
`(video, prompt)` media ref whose raw path is handed to the Qwen3-Omni processor
(it samples frames itself, fps-driven, so its TMRoPE grid/temporal metadata stays
consistent). The reward is `mc_exact_match` with `graded_format_reward`:
`<answer>X</answer>` scores 1.0, any other correct extraction 0.5, wrong 0.0.
