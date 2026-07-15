#!/usr/bin/env bash
# run_vllm_omni.sh — Launch Qwen3-Omni Thinker GSPO+LoRA training on the vLLM-Omni
# rollout engine.
#
# Recipe: examples/ar/qwen3_omni_video_r1_gspo_lora_vllm_omni_1x8.yaml
#   Anchored rollout: a TP=4 vLLM-Omni engine on 4 GPUs; training runs FSDP DP=8
#   across all 8 GPUs; sleep/wake time-shares the 4 rollout cards with training.
#
# Extra Hydra overrides may be passed through, e.g. for a quick check:
#   bash run_vllm_omni.sh num_rollouts=3 eval_interval=0 logging.report_to_wandb=false

set -uo pipefail

export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR=${WANDB_DIR:-./outputs/wandb_runs}

timestamp=$(date +%Y%m%d%H%M%S)

# Paths (override via env).
export QWEN3_OMNI_PATH=${QWEN3_OMNI_PATH:-/dev/shm/Qwen3-Omni-30B-A3B-Instruct}
export DATA_PATH=${DATA_PATH:-datasets/video_r1_260k/train.jsonl}
export EVAL_DATA_PATH=${EVAL_DATA_PATH:-datasets/video_r1_260k/val.jsonl}

# Conda env holding vllm-omni (override via env; skipped if conda is not present).
CONDA_ROOT=${CONDA_ROOT:-/opt/conda}
ENV_NAME=${ENV_NAME:-unirl-omni}

log() { printf '\033[1;36m[run-vllm-omni]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[run-vllm-omni ERROR]\033[0m %s\n' "$*" >&2; exit 1; }

# Pre-flight.
[ -d "$QWEN3_OMNI_PATH" ] || die "model dir missing: $QWEN3_OMNI_PATH"
[ -f "$DATA_PATH" ]       || die "train jsonl missing: $DATA_PATH"
[ -f "$EVAL_DATA_PATH" ]  || die "eval jsonl missing: $EVAL_DATA_PATH"

# Activate the conda env if one is available; otherwise assume it is already active.
if [ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "$CONDA_ROOT/etc/profile.d/conda.sh"
  conda activate "$ENV_NAME" || die "failed to activate conda env: $ENV_NAME"
fi

log "python:      $(python -V 2>&1)"
log "model:       $QWEN3_OMNI_PATH"
log "train / val: $DATA_PATH / $EVAL_DATA_PATH"

# HF offline — the model is on disk.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p outputs

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7} \
PYTHONPATH=$(pwd) \
python -m unirl.train_ar \
  --config-name=ar/qwen3_omni_video_r1_gspo_lora_vllm_omni_1x8 \
  "$@" \
  |& tee "outputs/qwen3_omni_vllm_omni_${timestamp}.log"
