#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
ENV_NAME=${ENV_NAME:-unirl-veomni-bitwise}
UNIMATCH_ROOT=${UNIMATCH_ROOT:-/root/bruceszchen_gy2/MyProjects/UniMatch}
RESPONSE_LENGTH=${RESPONSE_LENGTH:-1024}
export RESPONSE_LENGTH

source /opt/conda/etc/profile.d/conda.sh
conda activate "${ENV_NAME}"

# This signoff topology is deliberately pinned to physical H20 GPUs 4-7.
export CUDA_VISIBLE_DEVICES=4,5,6,7
export UNIRL_PHYSICAL_GPU_BASE=4
export QWEN3_MOE_PATH=${QWEN3_MOE_PATH:-/dev/shm/Qwen3-30B-A3B}
export PYTHONPATH="${REPO_ROOT}:${UNIMATCH_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export RAY_raylet_start_wait_time_s=${RAY_raylet_start_wait_time_s:-180}

export VLLM_PLUGINS=${VLLM_PLUGINS:-unimatch}
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_BATCH_INVARIANT=0
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1800}
export FLASH_ATTENTION_DETERMINISTIC=1
export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:16:8}

export UNIRL_EP_COMM_BACKEND=${UNIRL_EP_COMM_BACKEND:-unimatch}
export UNIRL_K3_DEBUG=${UNIRL_K3_DEBUG:-1}
export UNIRL_WEIGHT_DIGEST_VERIFY=${UNIRL_WEIGHT_DIGEST_VERIFY:-1}
export UNIRL_VEOMNI_STREAMING_GRAD_NORM=${UNIRL_VEOMNI_STREAMING_GRAD_NORM:-1}
export UNIMATCH_VEOMNI_EP4_EXACT=1
export UNIMATCH_STRICT_PATCH=1
export UNIMATCH_PATCHES=${UNIMATCH_PATCHES:-qwen3_moe,batch_invariant_reductions,batch_invariant_norm,batch_invariant_precision,batch_invariant_attention}
export UNIMATCH_REDUCTION_PROVIDER=${UNIMATCH_REDUCTION_PROVIDER:-vllm_bi}
export UNIMATCH_FSDP_PATCHES=qwen3_moe
export UNIMATCH_HF_ATEN_PATCHES=linear,softmax
export UNIMATCH_QWEN3_MOE_COMPONENTS=${UNIMATCH_QWEN3_MOE_COMPONENTS:-experts_grouped,qkv_local,oproj_col,lm_head_col,gate_fp32,router_hf}
export UNIMATCH_TRAIN_TP=1
export UNIMATCH_TRAIN_EP_SIZE=4
export UNIMATCH_DENSE_PROVIDER=${UNIMATCH_DENSE_PROVIDER:-vllm_bi}
export UNIMATCH_GROUPED_PROVIDER=${UNIMATCH_GROUPED_PROVIDER:-vllm_bi_loop}
export UNIMATCH_GROUPED_VARIANT=tma
export UNIMATCH_DENSE_VARIANT=tma
export UNIMATCH_MEGATRON=0
export UNIMATCH_MEGATRON_MOE_SUM_ORDER=none
export UNIMATCH_MOE_SUM_ORDER=none
export UNIMATCH_MOE_SUM=bf16
export UNIMATCH_MOE_COMBINE_PROVIDER=${UNIMATCH_MOE_COMBINE_PROVIDER:-torch}
export UNIMATCH_HF_ROUTER_SOFTMAX_ATEN=1
export UNIMATCH_ROUTER_SOFTMAX_IMPL=vllm_kernel
export UNIMATCH_BATCH_INVARIANT_NORM=1
export UNIMATCH_RMSNORM_PROVIDER=${UNIMATCH_RMSNORM_PROVIDER:-vllm_bi}
export UNIMATCH_QWEN3_ROPE_CONTRACT=1
export UNIMATCH_QWEN3_OPROJ_CONSISTENT=0
export UNIMATCH_ATTENTION_BACKEND=fa3

# Fail-closed UniMatch DeepEP-HT contract: EP4, 128 global / 32 local experts,
# BF16, intranode HT, and synchronous finish for correctness signoff.
export UNIMATCH_DEEPEP_EP_SIZE=4
export UNIMATCH_DEEPEP_DTYPE=bfloat16
export UNIMATCH_DEEPEP_SINGLE_NODE=1
export UNIMATCH_DEEPEP_MODE=ht
export UNIMATCH_DEEPEP_NUM_EXPERTS=128
export UNIMATCH_DEEPEP_LOCAL_EXPERTS=32
export UNIMATCH_DEEPEP_ASYNC_FINISH=0
export UNIMATCH_DEEPEP_ORACLE=0

# The native deepep_ht A/B path uses the equivalent UniRL-prefixed contract.
export UNIRL_DEEPEP_SINGLE_NODE=1
export UNIRL_DEEPEP_MODE=ht
export UNIRL_DEEPEP_ASYNC_FINISH=0

if [[ ! -f "${UNIMATCH_ROOT}/unimatch/adaptor/veomni/bootstrap.py" ]]; then
    echo "Missing UniMatch VeOmni adaptor: ${UNIMATCH_ROOT}/unimatch/adaptor/veomni/bootstrap.py" >&2
    exit 2
fi

export GSM8K_PARQUET_DIR=${GSM8K_PARQUET_DIR:-/apdcephfs_gy2/share_303407316/hunyuan/bruceszchen/HF_Models/hub/datasets--openai--gsm8k/verl_parquet}
export DATA_PATH=${DATA_PATH:-${REPO_ROOT}/datasets/gsm8k/train.jsonl}
export EVAL_DATA_PATH=${EVAL_DATA_PATH:-${REPO_ROOT}/datasets/gsm8k/test.jsonl}

if [[ ! -f "${DATA_PATH}" || ! -f "${EVAL_DATA_PATH}" ]]; then
    mkdir -p "$(dirname "${DATA_PATH}")" "$(dirname "${EVAL_DATA_PATH}")"
    python - <<'PY'
import json
import os

import pandas as pd

root = os.environ["GSM8K_PARQUET_DIR"]
for split, output in (("train", os.environ["DATA_PATH"]), ("test", os.environ["EVAL_DATA_PATH"])):
    frame = pd.read_parquet(os.path.join(root, f"{split}.parquet"))
    with open(output, "w", encoding="utf-8") as stream:
        for index, row in frame.iterrows():
            stream.write(
                json.dumps(
                    {
                        "prompt": row["prompt"][0]["content"],
                        "prompt_id": f"gsm8k-{split}-{index}",
                        "metadata": {"answer": str(row["reward_model"]["ground_truth"])},
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
PY
fi

python - <<'PY'
import importlib.util
import json
import os

backend = os.environ["UNIRL_EP_COMM_BACKEND"].strip().lower()
if backend not in {"torch", "deepep_ht", "unimatch"}:
    raise SystemExit(f"invalid UNIRL_EP_COMM_BACKEND={backend!r}")
if backend == "unimatch" and importlib.util.find_spec("unimatch.adaptor.veomni.bootstrap") is None:
    raise SystemExit("unimatch.adaptor.veomni.bootstrap is not importable")
print(
    "[unirl.veomni.bitwise] manifest="
    + json.dumps(
        {
            "physical_gpus": [4, 5, 6, 7],
            "actor": {"world": 4, "dp": 4, "ep": 4, "ep_fsdp": 1},
            "rollout": {"groups": 4, "tp": 1, "ep": 1, "sleep_mode": True},
            "ep_comm_backend": backend,
            "response_length": int(os.environ.get("RESPONSE_LENGTH", "1024")),
            "ignore_eos": True,
            "weight_sync": "TensorWeightSync",
        },
        sort_keys=True,
    ),
    flush=True,
)
PY

export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/urbit-veomni-$$}
RAY_PORT=${RAY_PORT:-27379}
RAY_MIN_WORKER_PORT=${RAY_MIN_WORKER_PORT:-54010}
RAY_MAX_WORKER_PORT=${RAY_MAX_WORKER_PORT:-54999}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-55002}
NODE_IP=${NODE_IP:-127.0.0.1}

cleanup() {
    for _attempt in 1 2 3; do
        while read -r pid; do
            [[ -n "${pid}" && "${pid}" != "$$" ]] || continue
            if [[ "${_attempt}" == "3" ]]; then
                kill -9 "${pid}" >/dev/null 2>&1 || true
            else
                kill "${pid}" >/dev/null 2>&1 || true
            fi
        done < <(pgrep -f "${RAY_TMPDIR}" || true)
        sleep 1
    done
}
trap cleanup EXIT

mkdir -p "${RAY_TMPDIR}" "${REPO_ROOT}/logs"
ray start --head \
    --node-ip-address="${NODE_IP}" \
    --port="${RAY_PORT}" \
    --dashboard-port="${RAY_DASHBOARD_PORT}" \
    --min-worker-port="${RAY_MIN_WORKER_PORT}" \
    --max-worker-port="${RAY_MAX_WORKER_PORT}" \
    --temp-dir="${RAY_TMPDIR}" \
    --num-cpus=16 \
    --num-gpus=4 \
    --include-dashboard=false \
    --disable-usage-stats >/dev/null
export RAY_ADDRESS="${NODE_IP}:${RAY_PORT}"

cd "${REPO_ROOT}"
run_id=$(date +%Y%m%d_%H%M%S)
log_file="logs/qwen3_moe_veomni_ep4_vllm_tp1_bitwise_${run_id}.log"
python -m unirl.train_ar \
    --config-name=ar/qwen3_moe_grpo_30b_a3b_gsm8k_veomni_ep4_vllm_tp1_bitwise \
    num_devices=4 \
    devices_per_node=4 \
    batch_size=4 \
    backend.fsdp_cfg.ep_size=4 \
    rollout.ep_size=1 \
    rollout.config.tp_size=1 \
    rollout.config.max_new_tokens="${RESPONSE_LENGTH}" \
    rollout.config.ignore_eos=true \
    rollout.config.engine_kwargs.max_num_seqs=1 \
    data_source.args.algorithm.prompts_per_rollout=4 \
    sampling.samples_per_prompt=1 \
    sampling.max_new_tokens="${RESPONSE_LENGTH}" \
    algorithm.horizon="${RESPONSE_LENGTH}" \
    algorithm.alignment_probe=true \
    algorithm.alignment_require_exact=true \
    "$@" \
    2>&1 | tee "${log_file}"
