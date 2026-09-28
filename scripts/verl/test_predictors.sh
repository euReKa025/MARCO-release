#!/usr/bin/env bash
set -euo pipefail

print_usage() {
  cat <<'EOF'
Usage: test_predictors.sh [--help|-h]

Starts or reuses predictor services, sends one smoke-test request to each service,
and prints the JSON responses.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  print_usage
  exit 0
fi

if [[ "$#" -gt 0 ]]; then
  echo "error: unknown argument: $1" >&2
  print_usage >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ -f "$ROOT_DIR/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$ROOT_DIR/.venv/bin/activate"
fi

SMILES="${PREDICTOR_TEST_SMILES:-CCO}"
TIMEOUT_S="${PREDICTOR_SMOKE_TIMEOUT_S:-60}"
START_PREDICTORS="${START_PREDICTORS:-1}"

if [[ "$START_PREDICTORS" == "1" ]]; then
  PREDICTOR_EXPORTS="$(bash "$ROOT_DIR/scripts/verl/start_predictors.sh")"
  eval "$PREDICTOR_EXPORTS"
fi

"$PYTHON_BIN" - "$SMILES" "$TIMEOUT_S" <<'PY'
import json
import os
import sys

import requests

smiles = sys.argv[1]
timeout_s = float(sys.argv[2])

checks = (
    ("ADMET", os.environ["MARCO_ADMET_API"], ("bbbp", "hia", "mutagenicity", "qed", "plogp")),
    ("DRD2", os.environ["MARCO_DRD2_API"], ("drd2",)),
)

session = requests.Session()
session.trust_env = False
for service_name, endpoint, expected_keys in checks:
    response = session.post(endpoint, json={"smiles": smiles}, timeout=timeout_s)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise SystemExit(f"{service_name} returned non-dict JSON: {type(payload).__name__}")
    missing = [key for key in expected_keys if key not in payload]
    if missing:
        raise SystemExit(f"{service_name} response missing expected keys {missing}: {payload}")
    print(
        json.dumps(
            {
                "service": service_name,
                "endpoint": endpoint,
                "smiles": smiles,
                "payload": payload,
            },
            ensure_ascii=False,
        )
    )
PY
