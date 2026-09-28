#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

INPUT_JSON="${INPUT_JSON:?INPUT_JSON is required}"
PROPERTY_SETTING="${PROPERTY_SETTING:-bbbp+drd2+plogp}"
SEEN_SETTING="${SEEN_SETTING:-seen}"
IND_SETTING="${IND_SETTING:-IND}"
EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-marco}"
METHOD_NAME="${METHOD_NAME:-marco}"
OUTPUT_FOLDER="${OUTPUT_FOLDER:-${ROOT_DIR}/outputs/standalone_eval/manual_metrics/${EXPERIMENT_PREFIX}_${METHOD_NAME}}"
SIMILARITY_THRESHOLD="${SIMILARITY_THRESHOLD:-0.4}"

python -m marco.eval.multi_turn_ckpt_evaluate \
  --input_json "$INPUT_JSON" \
  --property_setting "$PROPERTY_SETTING" \
  --seen_setting "$SEEN_SETTING" \
  --IND_setting "$IND_SETTING" \
  --experiment_prefix "$EXPERIMENT_PREFIX" \
  --method_name "$METHOD_NAME" \
  --output_folder "$OUTPUT_FOLDER" \
  --similarity_threshold "$SIMILARITY_THRESHOLD"
