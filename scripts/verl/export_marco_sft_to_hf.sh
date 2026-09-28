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

resolve_latest_global_step_dir() {
  local checkpoint_root="$1"
  local latest=""
  local base=""

  while IFS= read -r candidate; do
    base="$(basename "$candidate")"
    if [[ "$base" =~ ^global_step_[0-9]+$ ]]; then
      latest="$candidate"
    fi
  done < <(find "$checkpoint_root" -mindepth 1 -maxdepth 1 -type d -name 'global_step_[0-9]*' | sort -V)

  if [[ -z "$latest" ]]; then
    return 1
  fi
  printf '%s\n' "$latest"
}

resolve_fsdp_actor_dir() {
  local checkpoint_dir="$1"
  local candidate=""

  if [[ "$checkpoint_dir" == */actor/huggingface ]]; then
    candidate="$(dirname "$checkpoint_dir")"
  elif [[ "$checkpoint_dir" == */actor ]]; then
    candidate="$checkpoint_dir"
  elif [[ -d "$checkpoint_dir/actor" ]]; then
    candidate="$checkpoint_dir/actor"
  else
    candidate="$checkpoint_dir"
  fi

  if [[ ! -f "$candidate/fsdp_config.json" ]]; then
    echo "error: FSDP actor dir missing fsdp_config.json: $candidate" >&2
    return 1
  fi
  if [[ ! -f "$candidate/huggingface/config.json" ]]; then
    echo "error: FSDP actor dir missing huggingface/config.json: $candidate" >&2
    return 1
  fi

  printf '%s\n' "$candidate"
}

CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-}"
OUTPUT_HF_DIR="${OUTPUT_HF_DIR:-$ROOT_DIR/outputs/verl_sft/marco_qwen_hf}"
MERGE_LOCAL_DIR=""

if [[ -n "$CHECKPOINT_DIR" && "$CHECKPOINT_DIR" != /* ]]; then
  CHECKPOINT_DIR="$ROOT_DIR/$CHECKPOINT_DIR"
fi
if [[ -n "$CHECKPOINT_ROOT" && "$CHECKPOINT_ROOT" != /* ]]; then
  CHECKPOINT_ROOT="$ROOT_DIR/$CHECKPOINT_ROOT"
fi
if [[ "$OUTPUT_HF_DIR" != /* ]]; then
  OUTPUT_HF_DIR="$ROOT_DIR/$OUTPUT_HF_DIR"
fi

if [[ -z "$CHECKPOINT_DIR" ]]; then
  if [[ -z "$CHECKPOINT_ROOT" ]]; then
    CHECKPOINT_ROOT="$ROOT_DIR/outputs/verl_sft/marco_qwen"
  fi
  if [[ ! -d "$CHECKPOINT_ROOT" ]]; then
    echo "error: checkpoint root not found: $CHECKPOINT_ROOT" >&2
    exit 1
  fi
  if ! CHECKPOINT_DIR="$(resolve_latest_global_step_dir "$CHECKPOINT_ROOT")"; then
    echo "error: no numeric global_step_<n> checkpoint found under: $CHECKPOINT_ROOT" >&2
    exit 1
  fi
fi

if [[ ! -d "$CHECKPOINT_DIR" ]]; then
  echo "error: checkpoint dir not found: $CHECKPOINT_DIR" >&2
  exit 1
fi
if ! MERGE_LOCAL_DIR="$(resolve_fsdp_actor_dir "$CHECKPOINT_DIR")"; then
  exit 1
fi

mkdir -p "$OUTPUT_HF_DIR"
echo "[export] checkpoint_dir=$CHECKPOINT_DIR"
echo "[export] merger_local_dir=$MERGE_LOCAL_DIR"
echo "[export] output_hf_dir=$OUTPUT_HF_DIR"

python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "$MERGE_LOCAL_DIR" \
  --target_dir "$OUTPUT_HF_DIR" \
  --trust-remote-code \
  --use_cpu_initialization \
  "$@"
