#!/usr/bin/env bash
set -euo pipefail

print_usage() {
  cat <<'EOF'
Usage: start_predictors.sh [--help|-h]

Starts predictor service setup for MARCO verl workflows and prints shell exports.
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

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

START_SERVERS="${START_SERVERS:-1}"
ALLOW_EXISTING_SERVERS="${ALLOW_EXISTING_SERVERS:-1}"
RESTART_UNHEALTHY_PREDICTORS="${RESTART_UNHEALTHY_PREDICTORS:-1}"
MARCO_MOCK_PREDICTOR="${MARCO_MOCK_PREDICTOR:-0}"
DEFAULT_CONDA_MUMO_ENV="mumo"
if [[ -z "${CONDA_MUMO_ENV:-}" ]]; then
  if [[ -d "$DEFAULT_CONDA_MUMO_ENV" ]]; then
    CONDA_MUMO_ENV="$DEFAULT_CONDA_MUMO_ENV"
  else
    CONDA_MUMO_ENV="mumo"
  fi
fi
CONDA_ROOT="${CONDA_ROOT:-/opt/anaconda3}"
ADMET_PORT="${ADMET_PORT:-10086}"
DRD2_PORT="${DRD2_PORT:-10087}"
PREDICTOR_HOST="${PREDICTOR_HOST:-127.0.0.1}"
PREDICTOR_WAIT_TIMEOUT_S="${PREDICTOR_WAIT_TIMEOUT_S:-180}"
PREDICTOR_WAIT_POLL_INTERVAL_S="${PREDICTOR_WAIT_POLL_INTERVAL_S:-1.0}"
PREDICTOR_PROBE_REQUEST_TIMEOUT_S="${PREDICTOR_PROBE_REQUEST_TIMEOUT_S:-30}"
PREDICTOR_REUSE_TIMEOUT_S="${PREDICTOR_REUSE_TIMEOUT_S:-15}"
PREDICTOR_SHUTDOWN_TIMEOUT_S="${PREDICTOR_SHUTDOWN_TIMEOUT_S:-10}"
PREDICTOR_PROBE_SMILES="${PREDICTOR_PROBE_SMILES:-CCO}"
PREDICTOR_REQUIRE_SEMANTIC_PROBE="${PREDICTOR_REQUIRE_SEMANTIC_PROBE:-1}"
POST_SERVER_READY_SLEEP="${POST_SERVER_READY_SLEEP:-0}"
PREDICTOR_LOG_DIR="${PREDICTOR_LOG_DIR:-$ROOT_DIR/logs/predictors}"
PREDICTOR_PID_FILE="${PREDICTOR_PID_FILE:-}"

DEFAULT_PREDICTOR_CODE_ROOT="$ROOT_DIR/third_party/repo_predictors"
FALLBACK_REPO_ROOT="$(cd "${ROOT_DIR}/.." && pwd)/RePO"
PREDICTOR_CODE_ROOT="${PREDICTOR_CODE_ROOT:-$DEFAULT_PREDICTOR_CODE_ROOT}"
if [[ ! -d "$PREDICTOR_CODE_ROOT/multiprop_utils" && -d "$FALLBACK_REPO_ROOT/multiprop_utils" ]]; then
  PREDICTOR_CODE_ROOT="$FALLBACK_REPO_ROOT"
fi

ADMET_LOG="${PREDICTOR_LOG_DIR}/admet.log"
DRD2_LOG="${PREDICTOR_LOG_DIR}/drd2.log"
ADMET_EXPECTED_KEYS=(mutagenicity bbbp hia qed plogp)
DRD2_EXPECTED_KEYS=(drd2)

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" >&2
}

record_started_service_pid() {
  local name="$1"
  local pid="$2"

  if [[ -z "$PREDICTOR_PID_FILE" ]]; then
    return 0
  fi

  mkdir -p "$(dirname "$PREDICTOR_PID_FILE")"
  printf '%s\t%s\t%s\n' "$pid" "$name" "$(date '+%F %T')" >>"$PREDICTOR_PID_FILE"
}

if ! command -v conda >/dev/null 2>&1 && [[ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  set +u
  # shellcheck source=/dev/null
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
  set -u
fi
if command -v conda >/dev/null 2>&1; then
  set +u
  eval "$(conda shell.bash hook)"
  set -u
fi

port_in_use() {
  local port="$1"
  python -m marco.verl.services.predictors port-in-use --port "$port" >/dev/null 2>&1
}

endpoint_host() {
  local endpoint="$1"
  python - "$endpoint" <<'PY'
import sys
from urllib.parse import urlparse

endpoint = str(sys.argv[1] or "").strip()
if "://" not in endpoint:
    endpoint = f"http://{endpoint}"
parsed = urlparse(endpoint)
print(parsed.hostname or "127.0.0.1", end="")
PY
}

endpoint_port() {
  local endpoint="$1"
  python - "$endpoint" <<'PY'
import sys
from urllib.parse import urlparse

endpoint = str(sys.argv[1] or "").strip()
if "://" not in endpoint:
    endpoint = f"http://{endpoint}"
parsed = urlparse(endpoint)
print("" if parsed.port is None else parsed.port, end="")
PY
}

wait_for_port() {
  local host="$1"
  local port="$2"
  local name="$3"
  local timeout_s="$4"

  for ((i=1; i<=timeout_s; i++)); do
    if python - "$host" "$port" <<'PY' >/dev/null 2>&1
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(1.0)
ok = s.connect_ex((host, port)) == 0
s.close()
raise SystemExit(0 if ok else 1)
PY
    then
      log "${name} is listening on ${host}:${port}"
      return 0
    fi
    sleep 1
  done

  echo "error: ${name} did not become ready on ${host}:${port} within ${timeout_s}s" >&2
  return 1
}

wait_for_predictor() {
  local endpoint="$1"
  local name="$2"
  shift 2
  local expected_keys=("$@")
  local expected_args=()
  local expected_key
  local host
  local port
  host="$(endpoint_host "$endpoint")"
  port="$(endpoint_port "$endpoint")"

  if [[ "$PREDICTOR_REQUIRE_SEMANTIC_PROBE" != "1" ]]; then
    wait_for_port "$host" "$port" "$name" "$PREDICTOR_WAIT_TIMEOUT_S"
    return 0
  fi

  for expected_key in "${expected_keys[@]}"; do
    expected_args+=(--expected-key "$expected_key")
  done
  if python -m marco.verl.services.predictors wait-predictor \
    --endpoint "$endpoint" \
    "${expected_args[@]}" \
    --timeout-s "$PREDICTOR_WAIT_TIMEOUT_S" \
    --poll-interval-s "$PREDICTOR_WAIT_POLL_INTERVAL_S" \
    --request-timeout-s "$PREDICTOR_PROBE_REQUEST_TIMEOUT_S" \
    --probe-smiles "$PREDICTOR_PROBE_SMILES" >/dev/null 2>&1; then
    log "${name} passed predictor probe at ${endpoint}"
    return 0
  fi
  echo "error: ${name} did not pass predictor probe at ${endpoint} within ${PREDICTOR_WAIT_TIMEOUT_S}s" >&2
  return 1
}

probe_existing_predictor() {
  local endpoint="$1"
  local name="$2"
  shift 2
  local expected_keys=("$@")
  local expected_args=()
  local expected_key
  local host
  local port
  host="$(endpoint_host "$endpoint")"
  port="$(endpoint_port "$endpoint")"

  if [[ "$PREDICTOR_REQUIRE_SEMANTIC_PROBE" != "1" ]]; then
    if wait_for_port "$host" "$port" "$name" "$PREDICTOR_REUSE_TIMEOUT_S"; then
      log "Reusing existing ${name} at ${endpoint}"
      return 0
    fi
    echo "error: port is occupied but ${name} is not reachable on ${host}:${port} within ${PREDICTOR_REUSE_TIMEOUT_S}s" >&2
    return 1
  fi

  for expected_key in "${expected_keys[@]}"; do
    expected_args+=(--expected-key "$expected_key")
  done
  if python -m marco.verl.services.predictors wait-predictor \
    --endpoint "$endpoint" \
    "${expected_args[@]}" \
    --timeout-s "$PREDICTOR_REUSE_TIMEOUT_S" \
    --poll-interval-s "$PREDICTOR_WAIT_POLL_INTERVAL_S" \
    --request-timeout-s "$PREDICTOR_PROBE_REQUEST_TIMEOUT_S" \
    --probe-smiles "$PREDICTOR_PROBE_SMILES" >/dev/null 2>&1; then
    log "Reusing existing ${name} at ${endpoint}"
    return 0
  fi
  echo "error: port is occupied but ${name} probe failed at ${endpoint} within ${PREDICTOR_REUSE_TIMEOUT_S}s" >&2
  return 1
}

wait_for_port_release() {
  local port="$1"
  local name="$2"
  local timeout_s="$3"
  local elapsed=0
  while (( elapsed < timeout_s )); do
    if ! port_in_use "$port"; then
      return 0
    fi
    sleep 1
    elapsed=$((elapsed + 1))
  done
  echo "error: ${name} still occupies port ${port} after ${timeout_s}s" >&2
  return 1
}

evict_known_service_processes() {
  local name="$1"
  local process_pattern="$2"
  local port="$3"

  log "Evicting unhealthy ${name} processes matching pattern: ${process_pattern}"
  pkill -TERM -f "$process_pattern" >/dev/null 2>&1 || true
  if wait_for_port_release "$port" "$name" "$PREDICTOR_SHUTDOWN_TIMEOUT_S"; then
    return 0
  fi

  log "${name} still holds port ${port} after SIGTERM; escalating to SIGKILL"
  pkill -KILL -f "$process_pattern" >/dev/null 2>&1 || true
  wait_for_port_release "$port" "$name" "$PREDICTOR_SHUTDOWN_TIMEOUT_S"
}

recover_unhealthy_predictor() {
  local name="$1"
  local entrypoint="$2"
  local log_file="$3"
  local process_pattern="$4"
  local port="$5"
  local endpoint="$6"
  shift 6
  local expected_keys=("$@")

  if [[ "$RESTART_UNHEALTHY_PREDICTORS" != "1" ]]; then
    echo "error: refusing to restart unhealthy ${name}; set RESTART_UNHEALTHY_PREDICTORS=1 to allow recovery" >&2
    return 1
  fi

  evict_known_service_processes "$name" "$process_pattern" "$port"
  if port_in_use "$port"; then
    echo "error: ${name} port ${port} remains occupied after evicting known processes; refusing to overwrite unknown process" >&2
    return 1
  fi

  start_service "$name" "$entrypoint" "$log_file"
  wait_for_predictor "$endpoint" "$name" "${expected_keys[@]}"
}

start_service() {
  local name="$1"
  local entrypoint="$2"
  local log_file="$3"
  local conda_env_python=""

  if [[ -n "${MUMO_PYTHON:-}" ]]; then
    nohup bash -lc "cd '${PREDICTOR_CODE_ROOT}' && exec '${MUMO_PYTHON}' '${entrypoint}'" >"$log_file" 2>&1 &
    record_started_service_pid "$name" "$!"
    log "Started ${name} with MUMO_PYTHON=${MUMO_PYTHON}"
    return 0
  fi

  if [[ -n "$CONDA_MUMO_ENV" ]]; then
    if [[ -d "$CONDA_MUMO_ENV" && -x "$CONDA_MUMO_ENV/bin/python" ]]; then
      conda_env_python="$CONDA_MUMO_ENV/bin/python"
      nohup bash -lc "cd '${PREDICTOR_CODE_ROOT}' && exec '${conda_env_python}' '${entrypoint}'" >"$log_file" 2>&1 &
      record_started_service_pid "$name" "$!"
      log "Started ${name} with CONDA_MUMO_ENV python ${conda_env_python}"
      return 0
    fi
    if [[ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
      nohup bash -lc "source '${CONDA_ROOT}/etc/profile.d/conda.sh' && eval \"\$(conda shell.bash hook)\" && conda activate '${CONDA_MUMO_ENV}' && cd '${PREDICTOR_CODE_ROOT}' && exec python '${entrypoint}'" >"$log_file" 2>&1 &
      record_started_service_pid "$name" "$!"
      log "Started ${name} with conda activate ${CONDA_MUMO_ENV}"
      return 0
    fi
    if command -v conda >/dev/null 2>&1; then
      nohup bash -lc "eval \"\$(conda shell.bash hook)\" && conda activate '${CONDA_MUMO_ENV}' && cd '${PREDICTOR_CODE_ROOT}' && exec python '${entrypoint}'" >"$log_file" 2>&1 &
      record_started_service_pid "$name" "$!"
      log "Started ${name} with conda activate ${CONDA_MUMO_ENV}"
      return 0
    fi
    echo "error: CONDA_MUMO_ENV is set but conda activation is unavailable" >&2
    return 1
  fi

  nohup bash -lc "cd '${PREDICTOR_CODE_ROOT}' && exec python '${entrypoint}'" >"$log_file" 2>&1 &
  record_started_service_pid "$name" "$!"
  log "Started ${name} with current python interpreter"
}

print_exports() {
  python -m marco.verl.services.predictors print-exports \
    --repo-root "$ROOT_DIR" \
    --admet-port "$ADMET_PORT" \
    --drd2-port "$DRD2_PORT" \
    --host "$PREDICTOR_HOST" \
    --mock-mode "$MARCO_MOCK_PREDICTOR"
}

mkdir -p "$PREDICTOR_LOG_DIR"

EXPORT_BLOCK="$(print_exports)"
eval "$EXPORT_BLOCK"
printf '%s\n' "$EXPORT_BLOCK"

if [[ "$MARCO_MOCK_PREDICTOR" == "1" ]]; then
  log "MARCO_MOCK_PREDICTOR=1, skip starting predictor services"
  exit 0
fi

if [[ "$START_SERVERS" != "1" ]]; then
  log "START_SERVERS=${START_SERVERS}, skip service startup"
  exit 0
fi

if [[ ! -f "${PREDICTOR_CODE_ROOT}/multiprop_utils/admetModel_api.py" ]]; then
  echo "error: cannot find ${PREDICTOR_CODE_ROOT}/multiprop_utils/admetModel_api.py" >&2
  exit 1
fi
if [[ ! -f "${PREDICTOR_CODE_ROOT}/multiprop_utils/drd2Model_api.py" ]]; then
  echo "error: cannot find ${PREDICTOR_CODE_ROOT}/multiprop_utils/drd2Model_api.py" >&2
  exit 1
fi

ADMET_IN_USE=0
DRD2_IN_USE=0
if port_in_use "$ADMET_PORT"; then
  ADMET_IN_USE=1
fi
if port_in_use "$DRD2_PORT"; then
  DRD2_IN_USE=1
fi

if [[ "$ALLOW_EXISTING_SERVERS" != "1" ]] && { [[ "$ADMET_IN_USE" == "1" ]] || [[ "$DRD2_IN_USE" == "1" ]]; }; then
  echo "error: predictor service ports are already in use (${ADMET_PORT}, ${DRD2_PORT})" >&2
  exit 1
fi

if [[ "$ADMET_IN_USE" == "0" ]]; then
  start_service "ADMET server" "multiprop_utils/admetModel_api.py" "$ADMET_LOG"
else
  if ! probe_existing_predictor "$MARCO_ADMET_API" "ADMET server" "${ADMET_EXPECTED_KEYS[@]}"; then
    recover_unhealthy_predictor \
      "ADMET server" \
      "multiprop_utils/admetModel_api.py" \
      "$ADMET_LOG" \
      "multiprop_utils/admetModel_api.py" \
      "$ADMET_PORT" \
      "$MARCO_ADMET_API" \
      "${ADMET_EXPECTED_KEYS[@]}"
  fi
fi

if [[ "$DRD2_IN_USE" == "0" ]]; then
  start_service "DRD2 server" "multiprop_utils/drd2Model_api.py" "$DRD2_LOG"
else
  if ! probe_existing_predictor "$MARCO_DRD2_API" "DRD2 server" "${DRD2_EXPECTED_KEYS[@]}"; then
    recover_unhealthy_predictor \
      "DRD2 server" \
      "multiprop_utils/drd2Model_api.py" \
      "$DRD2_LOG" \
      "multiprop_utils/drd2Model_api.py" \
      "$DRD2_PORT" \
      "$MARCO_DRD2_API" \
      "${DRD2_EXPECTED_KEYS[@]}"
  fi
fi

if [[ "$ADMET_IN_USE" == "0" ]]; then
  wait_for_predictor "$MARCO_ADMET_API" "ADMET server" "${ADMET_EXPECTED_KEYS[@]}"
fi
if [[ "$DRD2_IN_USE" == "0" ]]; then
  wait_for_predictor "$MARCO_DRD2_API" "DRD2 server" "${DRD2_EXPECTED_KEYS[@]}"
fi

if [[ "$POST_SERVER_READY_SLEEP" != "0" ]]; then
  log "Sleeping ${POST_SERVER_READY_SLEEP}s after predictor readiness"
  sleep "$POST_SERVER_READY_SLEEP"
fi
