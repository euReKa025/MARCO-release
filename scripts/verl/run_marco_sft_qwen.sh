#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -f "$ROOT_DIR/.venv/bin/activate" ]]; then
  echo "error: missing virtual environment activate script: $ROOT_DIR/.venv/bin/activate" >&2
  exit 1
fi
source "$ROOT_DIR/.venv/bin/activate"

export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/../verl:${PYTHONPATH:-}"

if [[ -z "${VERL_CONFIG_PATH:-}" ]]; then
  SIBLING_VERL_CONFIG="$ROOT_DIR/../verl/verl/trainer/config"
  if [[ -d "$SIBLING_VERL_CONFIG" ]]; then
    VERL_CONFIG_PATH="$SIBLING_VERL_CONFIG"
  else
    VERL_CONFIG_PATH="$(
      python -c "from pathlib import Path; import verl; print((Path(verl.__file__).resolve().parent / 'trainer' / 'config').as_posix())"
    )"
  fi
fi
if [[ -z "${VERL_CONFIG_PATH:-}" || ! -d "$VERL_CONFIG_PATH" ]]; then
  echo "error: unable to resolve VERL_CONFIG_PATH (set it explicitly or install local ../verl)" >&2
  exit 1
fi
export VERL_CONFIG_PATH

CONFIG_PATH="${CONFIG_PATH:-marco/verl/config/sft_marco_fsdp.yaml}"
if [[ "$CONFIG_PATH" != /* ]]; then
  CONFIG_PATH="$ROOT_DIR/$CONFIG_PATH"
fi
CONFIG_DIR="$(dirname "$CONFIG_PATH")"
CONFIG_BASENAME="$(basename "$CONFIG_PATH")"
CONFIG_NAME="${CONFIG_BASENAME%.yaml}"
CONFIG_NAME="${CONFIG_NAME%.yml}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"
MODEL_PATH="${MODEL_PATH%/}"
TRAIN_PARQUET="${TRAIN_PARQUET:-data/sft_recomputed/train.parquet}"
VAL_PARQUET="${VAL_PARQUET:-null}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/verl_sft/marco_qwen}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-4}"
MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-8192}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-null}"
OPTIM_LR="${OPTIM_LR:-1e-5}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-constant}"
LR_WARMUP_STEPS_RATIO="${LR_WARMUP_STEPS_RATIO:-0.03}"
OPTIM_WEIGHT_DECAY="${OPTIM_WEIGHT_DECAY:-0.01}"
MESSAGES_KEY="${MESSAGES_KEY:-messages}"
PROJECT_NAME="${PROJECT_NAME:-marco-verl}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-marco-sft}"
NNODES="${NNODES:-1}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-1}"
LOGGER="${LOGGER:-[console]}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
NODE_RANK="${NODE_RANK:-0}"

TORCHRUN_ARGS=(--nnodes "$NNODES" --nproc-per-node "$N_GPUS_PER_NODE")
if [[ "$NNODES" == "1" ]]; then
  TORCHRUN_ARGS+=(--standalone)
else
  TORCHRUN_ARGS+=(--node-rank "$NODE_RANK" --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT")
fi

"$TORCHRUN_BIN" "${TORCHRUN_ARGS[@]}" -m verl.trainer.sft_trainer \
  --config-path "$CONFIG_DIR" \
  --config-name "$CONFIG_NAME" \
  data.train_files="$TRAIN_PARQUET" \
  data.val_files="$VAL_PARQUET" \
  data.messages_key="$MESSAGES_KEY" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.micro_batch_size_per_gpu="$MICRO_BATCH_SIZE_PER_GPU" \
  data.max_token_len_per_gpu="$MAX_TOKEN_LEN_PER_GPU" \
  data.max_length="$MAX_LENGTH" \
  model.path="$MODEL_PATH" \
  trainer.default_local_dir="$OUTPUT_DIR" \
  trainer.project_name="$PROJECT_NAME" \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
  trainer.nnodes="$NNODES" \
  trainer.n_gpus_per_node="$N_GPUS_PER_NODE" \
  trainer.logger="$LOGGER" \
  optim.lr="$OPTIM_LR" \
  optim.lr_scheduler_type="$LR_SCHEDULER_TYPE" \
  optim.lr_warmup_steps_ratio="$LR_WARMUP_STEPS_RATIO" \
  optim.weight_decay="$OPTIM_WEIGHT_DECAY" \
  "$@"
