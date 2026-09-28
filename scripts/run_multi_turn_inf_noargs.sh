#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONDA_ROOT="${CONDA_ROOT:-/opt/anaconda3}"
DEFAULT_CONDA_MUMO_ENV="mumo"
CONDA_ENV="${CONDA_ENV:-${CONDA_MUMO_ENV:-}}"
if [[ -z "$CONDA_ENV" ]]; then
  if [[ -d "$DEFAULT_CONDA_MUMO_ENV" ]]; then
    CONDA_ENV="$DEFAULT_CONDA_MUMO_ENV"
  else
    CONDA_ENV="mumo"
  fi
fi
HF_CHECKPOINT_ROOT="${HF_CHECKPOINT_ROOT:-${ROOT_DIR}/outputs/marco_sft_think_qwen2_5_3b_hf}"
PROPERTY_SETTING="${PROPERTY_SETTING:-bbbp+drd2+plogp}"
SEEN_SETTING="${SEEN_SETTING:-seen}"
IND_SETTING="${IND_SETTING:-IND}"
METHOD_NAME="${METHOD_NAME:-marco}"
EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-marco}"
MODEL_PATH="${MODEL_PATH:-${HF_CHECKPOINT_ROOT}/${PROPERTY_SETTING//+/_}}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/${EXPERIMENT_PREFIX}_${METHOD_NAME}}"
OUTPUT_NAME="${OUTPUT_NAME:-processed_${IND_SETTING}_${SEEN_SETTING}_${PROPERTY_SETTING}_test_data.json}"
CANONICAL_DATA_DIR="${CANONICAL_DATA_DIR:-${ROOT_DIR}/data/canonical}"
PROMPT_MODE="${PROMPT_MODE:-think_answer}"
MAX_TURNS="${MAX_TURNS:-5}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-8}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-25}"
DO_SAMPLE="${DO_SAMPLE:-false}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
AUTO_START_SERVERS="${AUTO_START_SERVERS:-true}"
SERVER_READY_WAIT="${SERVER_READY_WAIT:-10}"
MARCO_VENV="${MARCO_VENV:-${ROOT_DIR}/.venv}"
PREDICTOR_REQUIRE_SEMANTIC_PROBE="${PREDICTOR_REQUIRE_SEMANTIC_PROBE:-1}"
RESTART_UNHEALTHY_PREDICTORS="${RESTART_UNHEALTHY_PREDICTORS:-1}"
PREDICTOR_LOG_DIR="${PREDICTOR_LOG_DIR:-${ROOT_DIR}/logs/multi_turn_eval_servers}"
PREDICTOR_PID_FILE="${PREDICTOR_PID_FILE:-${PREDICTOR_LOG_DIR}/predictor_pids.tsv}"
CLEANUP_EVAL_PREDICTORS="${CLEANUP_EVAL_PREDICTORS:-1}"

if [[ -f "${MARCO_VENV}/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${MARCO_VENV}/bin/activate"
fi

cleanup_predictors() {
  if [[ "${CLEANUP_EVAL_PREDICTORS}" != "1" && "${CLEANUP_EVAL_PREDICTORS}" != "true" ]]; then
    return 0
  fi
  if [[ ! -f "$PREDICTOR_PID_FILE" ]]; then
    return 0
  fi
  while IFS=$'\t' read -r pid _name _started_at; do
    if [[ -n "$pid" ]]; then
      pkill -TERM -P "$pid" 2>/dev/null || true
      kill "$pid" 2>/dev/null || true
    fi
  done <"$PREDICTOR_PID_FILE"
}
trap cleanup_predictors EXIT INT TERM

if [[ "${AUTO_START_SERVERS}" == "true" ]]; then
  mkdir -p "$PREDICTOR_LOG_DIR"
  : >"$PREDICTOR_PID_FILE"
  PREDICTOR_EXPORTS="$(
    CONDA_MUMO_ENV="$CONDA_ENV" \
    PREDICTOR_LOG_DIR="$PREDICTOR_LOG_DIR" \
    PREDICTOR_PID_FILE="$PREDICTOR_PID_FILE" \
    PREDICTOR_REQUIRE_SEMANTIC_PROBE="$PREDICTOR_REQUIRE_SEMANTIC_PROBE" \
    RESTART_UNHEALTHY_PREDICTORS="$RESTART_UNHEALTHY_PREDICTORS" \
    bash "$ROOT_DIR/scripts/verl/start_predictors.sh"
  )"
  eval "$PREDICTOR_EXPORTS"
  if [[ "${SERVER_READY_WAIT}" -gt 0 ]]; then
    sleep "${SERVER_READY_WAIT}"
  fi
fi

mkdir -p "${OUTPUT_DIR}"
SAMPLING_ARGS=()
if [[ "$DO_SAMPLE" == "1" || "$DO_SAMPLE" == "true" ]]; then
  SAMPLING_ARGS+=(--do_sample --temperature "$TEMPERATURE" --top_p "$TOP_P")
fi

python -m marco.eval.multi_turn_inference \
  --model_path "$MODEL_PATH" \
  --property_setting "$PROPERTY_SETTING" \
  --seen_setting "$SEEN_SETTING" \
  --IND_setting "$IND_SETTING" \
  --output_dir "$OUTPUT_DIR" \
  --output_name "$OUTPUT_NAME" \
  --canonical_data_dir "$CANONICAL_DATA_DIR" \
  --method_name "$METHOD_NAME" \
  --experiment_prefix "$EXPERIMENT_PREFIX" \
  --prompt_mode "$PROMPT_MODE" \
  --max_turns "$MAX_TURNS" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --generation_batch_size "$GENERATION_BATCH_SIZE" \
  --progress_interval "$PROGRESS_INTERVAL" \
  "${SAMPLING_ARGS[@]}"
