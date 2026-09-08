# experimental/train_inference_parity

Owner: UniRL Qwen3-MoE parity maintainers

This package incubates exact rollout-versus-old-policy log-probability
contracts without shipping monkey patches or experimental kernels in the UniRL
wheel.

## Scope

The first model profile is `Qwen3-30B-A3B`:

```text
experimental model id: qwen3_moe_30b_a3b
actor scoring: full prompt+response no-grad forward
comparison dtype: FP32
required metrics: max_absdiff=0, K3 mean/max=0, torch.equal=true
```

Supported topologies:

- FSDP actor world=4 with one vLLM TP4 rollout
- VeOmni DP4/EP4 actor with four vLLM TP1 rollouts and DeepEP HT

## Install

Install the normal UniRL vLLM/VeOmni environment, then install the experimental
plugin metadata and package:

```bash
pip install -e experimental/train_inference_parity/vllm_plugin
```

The installed distribution is `unirl-train-inference-parity-vllm`; its vLLM
general-plugin entry point is `unirl_train_inference_parity`.

## Launch

A Ray cluster must already be running. From the repository root:

```bash
export QWEN3_MOE_PATH=/dev/shm/Qwen3-30B-A3B
export DATA_PATH=/tmp/unirl_gsm8k_4_7.jsonl
export EVAL_DATA_PATH=/tmp/unirl_gsm8k_4_7.jsonl

python -m experimental.train_inference_parity.run \
  --config-name=qwen3_moe_30b_a3b_fsdp_tp4
```

VeOmni EP4:

```bash
export UNIRL_DEEPEP_MODE=ht
export UNIRL_DEEPEP_SINGLE_NODE=1
export UNIRL_DEEPEP_ASYNC_FINISH=0

python -m experimental.train_inference_parity.run \
  --config-name=qwen3_moe_30b_a3b_veomni_ep4_tp1
```

Hydra overrides can enable scheduler features:

```bash
rollout.config.engine_kwargs.enable_chunked_prefill=true
rollout.config.engine_kwargs.enable_prefix_caching=true
```

## Profiles

`public_reference` is the default:

- vLLM batch-invariant linear, norm, reductions and softmax
- vLLM native FA3 and MoE permute
- Torch fixed-order EP combine and backward recompute
- DeepEP HT and NCCL collectives

`optimized` is intentionally unavailable until experiment-owned kernels have a
reviewed Apache-2.0-compatible provenance. Setting it without
`UNIRL_PARITY_OPTIMIZED_LICENSE_ACCEPTED=1` fails before model construction.

## Package boundaries

- `unirl/` never imports this package.
- Recipes under this package may target `experimental.train_inference_parity.*`.
- The nested vLLM plugin distribution does not import `unirl` or sibling
  experimental packages.
- Common code needed by a second experimental package must graduate into core;
  it is not imported sideways.

## Diagnostics

Set:

```bash
export UNIRL_PARITY_DUMP_DIR=/path/to/tensor_dump
export UNIRL_PARITY_DUMP_RUN_ID=my_run
```

The actor and vLLM plugin write rank/layer/call tensor files beneath that root.

## Verification

| Profile | Topology | Response | Chunked prefill | Prefix cache | Update/reload | Status |
|---|---|---:|---:|---:|---:|---|
| public_reference | FSDP4 / vLLM TP4 | 1024 | on | on | two steps + reload | PASS |
| public_reference | VeOmni EP4 / 4×TP1 | 1024 | on, budget=128 | on | two steps + reload + 4-replica digest | PASS |
| optimized | both | 1024 | — | — | — | blocked on provenance/license |

Both PASS rows were run from this experimental package with
`VLLM_PLUGINS=unirl_train_inference_parity` and no external UniMatch plugin.
Every rank reported:

```text
token_count=4096
torch_equal=True
mismatch_count=0
max_absdiff_fp32=0
k3_mean=0
k3_max=0
```

FSDP/TP4:

```text
rollout 1: grad_norm=0.6523
rollout 2: grad_norm=1.0078
```

VeOmni EP4:

```text
rollout 1: grad_norm=1.7114
updated replicas=4 digest=c066983de34fb0f1f35b70e299c97676521123d8a59c1f5298dca0a7f601bd70
rollout 2: grad_norm=1.9102
```

During migration, EP4 initially differed by one FP32 ULP because the parity
topology variables were set after connecting to an existing Ray cluster.
`run.py` now creates a Ray job runtime environment before any role is spawned,
so `UNIRL_PARITY_TRAIN_EP_SIZE=4` reaches actor and rollout workers
consistently.

The repository wheel was also built and inspected: 595 wheel entries, zero
`experimental/` or `train_inference_parity` entries.
