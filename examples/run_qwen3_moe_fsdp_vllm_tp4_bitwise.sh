#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
ENV_NAME=${ENV_NAME:-unirl}

source /opt/conda/etc/profile.d/conda.sh
conda activate "${ENV_NAME}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export QWEN3_MOE_PATH=${QWEN3_MOE_PATH:-/dev/shm/Qwen3-30B-A3B}
export PYTHONPATH="${REPO_ROOT}:/root/bruceszchen_gy2/MyProjects/UniMatch${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export RAY_raylet_start_wait_time_s=${RAY_raylet_start_wait_time_s:-180}

export VLLM_PLUGINS=${VLLM_PLUGINS:-unimatch}
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_BATCH_INVARIANT=0
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1800}
export FLASH_ATTENTION_DETERMINISTIC=1
export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:16:8}

export UNIMATCH_STRICT_PATCH=1
export UNIMATCH_PATCHES=${UNIMATCH_PATCHES:-qwen3_moe,batch_invariant_reductions,batch_invariant_norm,batch_invariant_precision,batch_invariant_attention}
if [[ "${UNIMATCH_DISABLE_VLLM_PATCH:-0}" == "1" ]]; then
    export UNIMATCH_PATCHES=
fi
export UNIMATCH_FSDP_PATCHES=qwen3_moe
export UNIMATCH_HF_ATEN_PATCHES=linear,softmax
export UNIMATCH_QWEN3_MOE_COMPONENTS=${UNIMATCH_QWEN3_MOE_COMPONENTS:-experts_grouped,qkv_local,oproj_col,lm_head_col,gate_fp32,router_hf}
export UNIMATCH_TRAIN_TP=4
export UNIMATCH_TRAIN_EP_SIZE=1
export UNIMATCH_MEGATRON=0
export UNIMATCH_MEGATRON_MOE_SUM_ORDER=none
export UNIMATCH_MOE_SUM_ORDER=none
export UNIMATCH_MOE_SUM=bf16
export UNIMATCH_HF_ROUTER_SOFTMAX_ATEN=1
export UNIMATCH_ROUTER_SOFTMAX_IMPL=vllm_kernel
export UNIMATCH_BATCH_INVARIANT_NORM=1
export UNIMATCH_QWEN3_ROPE_CONTRACT=1
export UNIMATCH_QWEN3_OPROJ_CONSISTENT=0
export UNIMATCH_ATTENTION_BACKEND=fa3

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

export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/urbit$$}
RAY_PORT=${RAY_PORT:-26379}
RAY_MIN_WORKER_PORT=${RAY_MIN_WORKER_PORT:-53000}
RAY_MAX_WORKER_PORT=${RAY_MAX_WORKER_PORT:-53999}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-54002}
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
log_file="logs/qwen3_moe_fsdp_vllm_tp4_bitwise_${run_id}.log"
python -m unirl.train_ar \
    --config-name=ar/qwen3_moe_grpo_30b_a3b_gsm8k_fsdp_vllm_tp4_bitwise \
    "$@" 2>&1 | tee "${log_file}"
