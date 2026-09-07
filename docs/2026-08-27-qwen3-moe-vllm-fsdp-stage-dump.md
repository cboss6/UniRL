# Qwen3-MoE vLLM TP4 / FSDP stage dump

## Usage

Run the bitwise recipe with these additional variables:

```bash
export UNIMATCH_STAGE_DUMP_DIR=$PWD/logs/stage_dumps
export UNIMATCH_STAGE_DUMP_RUN_ID=my_run
export UNIMATCH_STAGE_DUMP_LAYERS=0
export UNIMATCH_STAGE_DUMP_CALLS=0-1
export UNIMATCH_STAGE_DUMP_RANKS=0-3
export UNIMATCH_STAGE_DUMP_STAGES=layer_input,attention_input,q_proj,k_proj,v_proj,q_norm,k_norm,q_rope,k_rope,v_attn,attn_core_output,oproj_input,oproj_output,router_logits,moe_input,moe_output
```

`call=0` is prompt prefill; later calls are one-token decode steps. Compare
rank-0 FSDP tensors with reconstructed TP4 vLLM tensors:

```bash
python examples/compare_qwen3_stage_dumps.py \
  --root logs/stage_dumps \
  --run-id my_run \
  --skip-stages layer_output \
  --output logs/stage_dumps/my_run/comparison.json
```

The comparator concatenates TP shards for Q/K/V, Q/K norm, RoPE, attention
core, and o-projection input. `layer_output` is skipped because vLLM keeps the
residual as a separate tensor while HF returns the residual-added state.

## 2026-08-27 findings

1. Prefill inputs, Q/K/V projections, and Q/K norm were exact. The first
   divergence was layer 0 `q_rope`, starting at position 1. UniMatch's vLLM
   adaptor had omitted the canonical eager RoPE contract used by the HF actor.
2. After restoring that contract, all 12 captured layer-0 prefill boundaries
   were exact and the one-token K3 became zero.
3. Decode call 1 had exact Q/K/V and RoPE inputs, but diverged at
   `attn_core_output`. The HF FA3 fixed-shape wrapper used the query sequence
   length for both Q and KV cumulative lengths. For `q_len=1, kv_len=17`, it
   exposed only one key to FA3.
4. Using independent Q and KV cumulative lengths made the 2-token and 8-token
   gates strictly exact (`K3=0`).
5. A 128-token probe then exposed a second attention boundary: contiguous and
   paged FA3 are exact through KV length 128 but differ from length 129 onward,
   even with `num_splits=1`. The FSDP no-grad one-token replay now packs K/V
   into 16-token pages and calls the same paged FA3 contract as vLLM.
6. After the paged-KV change, calls 112–114 (KV lengths 128–130) were exact
   across all 48 layers and 10 captured stages: 1,440 tensor comparisons,
   zero differing elements.
7. The full prompt-length-16 plus response-length-1024 gate completed with
   `mismatch_count=0` on every rank and trainer-reported `K3=0`.

Validated artifacts:

- `logs/stage_dumps/unirl_tp4_fsdp_20260827k/comparison.json`
- `logs/stage_dumps/unirl_tp4_fsdp_20260827l/comparison_decode.json`
- `logs/stage_dumps/unirl_tp4_fsdp_kv128_crossing_20260827/comparison.json`
- terminal gate: 8-token rollout at `2026-08-27 13:32:25`, `K3=0`
- `/tmp/unirl_paged_gate1024_20260827T092145Z.log`: 1024-token rollout at
  `2026-08-27 18:33:00`, `K3=0`
