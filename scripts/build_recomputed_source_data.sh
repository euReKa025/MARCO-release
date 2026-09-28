#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -f "$ROOT_DIR/.venv/bin/activate" ]]; then
  echo "error: missing virtual environment activate script: $ROOT_DIR/.venv/bin/activate" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "$ROOT_DIR/.venv/bin/activate"

export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/../verl:${PYTHONPATH:-}"

CANONICAL_SOURCE_DIR="${CANONICAL_SOURCE_DIR:-data/canonical_source_recomputed}"
SFT_THINK_DIR="${SFT_THINK_DIR:-data/sft_think}"
RLHF_PROMPT_MODE="${RLHF_PROMPT_MODE:-think_answer}"
VERIFY_PROMPT_ALIGNMENT="${VERIFY_PROMPT_ALIGNMENT:-1}"

python -m marco.data.build_recomputed_source_datasets \
  --canonical-source-dir "$CANONICAL_SOURCE_DIR" \
  --sft-think-dir "$SFT_THINK_DIR" \
  --canonical-output-dir "data/canonical_recomputed" \
  --baseline-jsonl-output-dir "data/baselines_recomputed" \
  --sft-aligned-output-dir "data/sft_think_aligned_recomputed" \
  --sft-parquet-output-dir "data/sft_recomputed" \
  --baseline-parquet-output-dir "data/grpo_single_turn_recomputed" \
  --rlhf-parquet-output-dir "data/rlhf_recomputed" \
  --rlhf-prompt-mode "$RLHF_PROMPT_MODE" \
  "$@"

if [[ "$VERIFY_PROMPT_ALIGNMENT" == "1" || "$VERIFY_PROMPT_ALIGNMENT" == "true" ]]; then
  python -m marco.data.verify_instruction_prompt_alignment \
    --canonical-dir "data/canonical_recomputed" \
    --rlhf-parquet-dir "data/rlhf_recomputed" \
    --sft-parquet-dir "data/sft_recomputed"
fi
