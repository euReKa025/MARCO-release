#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

VALIDATION_DIR="${TRAINER_VALIDATION_DATA_DIR:-}"
EVAL_DIR="${TRAINER_EVAL_ROLLOUT_DIR:-}"
POLL_SECONDS="${VALIDATION_EVAL_WATCH_POLL_SECONDS:-5}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TRAINER_EVAL_SUBTASK="${TRAINER_EVAL_SUBTASK:-}"
TRAINER_EVAL_PARTITION="${TRAINER_EVAL_PARTITION:-seen}"
TRAINER_EVAL_MODE="${TRAINER_EVAL_MODE:-sampled}"
TRAINER_RAW_EVAL_ROOT_DIR="${TRAINER_RAW_EVAL_ROOT_DIR:-$ROOT_DIR}"
TRAINER_RAW_EVAL_SIMILARITY_THRESHOLD="${TRAINER_RAW_EVAL_SIMILARITY_THRESHOLD:-0.5}"
TRAINER_RAW_EVAL_SIMILARITY_TARGET_LOW="${TRAINER_RAW_EVAL_SIMILARITY_TARGET_LOW:-}"
TRAINER_RAW_EVAL_SIMILARITY_TARGET_HIGH="${TRAINER_RAW_EVAL_SIMILARITY_TARGET_HIGH:-}"
TRAINER_RAW_EVAL_SIMILARITY_COPY_THRESHOLD="${TRAINER_RAW_EVAL_SIMILARITY_COPY_THRESHOLD:-}"
TRAINER_EVAL_RUN_METADATA_JSON="${TRAINER_EVAL_RUN_METADATA_JSON:-}"

if [[ -z "$VALIDATION_DIR" || -z "$EVAL_DIR" ]]; then
  exit 0
fi

mkdir -p "$EVAL_DIR"

extract_step() {
  local stem="$1"
  if [[ "$stem" =~ ^[0-9]+$ ]]; then
    printf '%s\n' "$stem"
    return
  fi
  printf '\n'
}

while true; do
  for jsonl in "$VALIDATION_DIR"/*.jsonl; do
    [[ -e "$jsonl" ]] || continue
    base_name="$(basename "$jsonl")"
    stem="${base_name%.jsonl}"
    step="$(extract_step "$stem")"
    if [[ -z "$step" ]]; then
      continue
    fi

    output_json="$EVAL_DIR/eval_rollout_${step}.json"
    raw_summary_json="$EVAL_DIR/raw_summary_${step}.json"
    if [[ -f "$output_json" && ! "$jsonl" -nt "$output_json" ]]; then
      if [[ -f "$raw_summary_json" && ! "$output_json" -nt "$raw_summary_json" ]]; then
        continue
      fi
    fi

    tmp_output="${output_json}.tmp.$$"
    ROLLOUT_ARGS=(
      -m marco.eval.build_validate_rollout_snapshot
      --input-jsonl "$jsonl"
      --output-json "$tmp_output"
      --step "$step"
      --run-metadata-from-env
    )
    if [[ -n "$TRAINER_EVAL_SUBTASK" ]]; then
      ROLLOUT_ARGS+=(--subtask "$TRAINER_EVAL_SUBTASK")
    fi
    if [[ -n "$TRAINER_EVAL_RUN_METADATA_JSON" ]]; then
      ROLLOUT_ARGS+=(--run-metadata-json "$TRAINER_EVAL_RUN_METADATA_JSON")
    fi
    if "$PYTHON_BIN" "${ROLLOUT_ARGS[@]}"; then
      tmp_raw_summary="${raw_summary_json}.tmp.$$"
      RAW_SUMMARY_ARGS=(
        -m marco.eval.build_validate_raw_summary
        --snapshot-json "$tmp_output"
        --output-json "$tmp_raw_summary"
        --partition "$TRAINER_EVAL_PARTITION"
        --validation-mode "$TRAINER_EVAL_MODE"
        --root-dir "$TRAINER_RAW_EVAL_ROOT_DIR"
        --similarity-threshold "$TRAINER_RAW_EVAL_SIMILARITY_THRESHOLD"
        --run-metadata-from-env
      )
      if [[ -n "$TRAINER_EVAL_SUBTASK" ]]; then
        RAW_SUMMARY_ARGS+=(--subtask "$TRAINER_EVAL_SUBTASK")
      fi
      if [[ -n "$TRAINER_RAW_EVAL_SIMILARITY_TARGET_LOW" ]]; then
        RAW_SUMMARY_ARGS+=(--similarity-target-low "$TRAINER_RAW_EVAL_SIMILARITY_TARGET_LOW")
      fi
      if [[ -n "$TRAINER_RAW_EVAL_SIMILARITY_TARGET_HIGH" ]]; then
        RAW_SUMMARY_ARGS+=(--similarity-target-high "$TRAINER_RAW_EVAL_SIMILARITY_TARGET_HIGH")
      fi
      if [[ -n "$TRAINER_RAW_EVAL_SIMILARITY_COPY_THRESHOLD" ]]; then
        RAW_SUMMARY_ARGS+=(--similarity-copy-threshold "$TRAINER_RAW_EVAL_SIMILARITY_COPY_THRESHOLD")
      fi
      if [[ -n "$TRAINER_EVAL_RUN_METADATA_JSON" ]]; then
        RAW_SUMMARY_ARGS+=(--run-metadata-json "$TRAINER_EVAL_RUN_METADATA_JSON")
      fi
      if "$PYTHON_BIN" "${RAW_SUMMARY_ARGS[@]}"; then
        mv "$tmp_raw_summary" "$raw_summary_json"
      else
        rm -f "$tmp_raw_summary"
      fi
      mv "$tmp_output" "$output_json"
    else
      rm -f "$tmp_output"
    fi
  done
  sleep "$POLL_SECONDS"
done
