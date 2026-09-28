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

REPO_DATA_ROOT="${REPO_DATA_ROOT:-../RePO/data}"
SFT_THINK_DIR="${SFT_THINK_DIR:-data/sft_think}"
RLHF_PROMPT_MODE="${RLHF_PROMPT_MODE:-think_answer}"

python -m marco.data.build_repo_source_datasets \
  --repo-data-root "$REPO_DATA_ROOT" \
  --sft-think-dir "$SFT_THINK_DIR" \
  --canonical-output-dir "data/canonical" \
  --baseline-jsonl-output-dir "data/baselines" \
  --sft-aligned-output-dir "data/sft_think_aligned" \
  --sft-parquet-output-dir "data/sft" \
  --baseline-parquet-output-dir "data/grpo_single_turn" \
  --rlhf-parquet-output-dir "data/rlhf" \
  --rlhf-prompt-mode "$RLHF_PROMPT_MODE" \
  "$@"
