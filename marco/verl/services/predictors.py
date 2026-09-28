from __future__ import annotations

import argparse
import os
import shlex
import socket
import time
from pathlib import Path
from typing import Mapping, Sequence
from urllib.parse import ParseResult, urlparse, urlunparse

import requests

DEFAULT_ADMET_PORT = 10086
DEFAULT_DRD2_PORT = 10087
DEFAULT_PREDICTOR_HOST = "127.0.0.1"
DEFAULT_ADMET_API = f"http://{DEFAULT_PREDICTOR_HOST}:{DEFAULT_ADMET_PORT}/predict/"
DEFAULT_DRD2_API = f"http://{DEFAULT_PREDICTOR_HOST}:{DEFAULT_DRD2_PORT}/predict/"
DEFAULT_PREDICTOR_CACHE_RELATIVE = Path("outputs/predictor_cache.sqlite3")
DEFAULT_PROBE_SMILES = "CCO"


def _env_map(environ: Mapping[str, str] | None = None) -> Mapping[str, str]:
    return os.environ if environ is None else environ


def _normalize_predict_endpoint(raw: str, *, default_port: int) -> str:
    text = raw.strip()
    if not text:
        return f"http://{DEFAULT_PREDICTOR_HOST}:{default_port}/predict/"
    if "://" not in text:
        text = f"http://{text}"

    parsed = urlparse(text)
    path = parsed.path or ""
    if path in ("", "/"):
        path = "/predict/"
    elif path.endswith("/predict"):
        path = f"{path}/"
    elif path.endswith("/predict/"):
        pass
    elif path.endswith("/"):
        path = f"{path}predict/"
    rebuilt = ParseResult(
        scheme=parsed.scheme or "http",
        netloc=parsed.netloc,
        path=path,
        params=parsed.params,
        query=parsed.query,
        fragment=parsed.fragment,
    )
    return urlunparse(rebuilt)


def predictor_endpoints(
    *,
    environ: Mapping[str, str] | None = None,
    admet_port: int = DEFAULT_ADMET_PORT,
    drd2_port: int = DEFAULT_DRD2_PORT,
    host: str = DEFAULT_PREDICTOR_HOST,
) -> dict[str, str]:
    env = _env_map(environ)
    default_admet = f"http://{host}:{int(admet_port)}/predict/"
    default_drd2 = f"http://{host}:{int(drd2_port)}/predict/"
    admet_raw = env.get("MARCO_ADMET_API", default_admet)
    drd2_raw = env.get("MARCO_DRD2_API", default_drd2)
    return {
        "MARCO_ADMET_API": _normalize_predict_endpoint(admet_raw, default_port=int(admet_port)),
        "MARCO_DRD2_API": _normalize_predict_endpoint(drd2_raw, default_port=int(drd2_port)),
    }


def predictor_cache_path(
    *,
    repo_root: str | Path,
    environ: Mapping[str, str] | None = None,
) -> Path:
    env = _env_map(environ)
    root = Path(repo_root).resolve()
    raw = env.get("MARCO_PREDICTOR_CACHE_PATH", "").strip()
    if not raw:
        return root / DEFAULT_PREDICTOR_CACHE_RELATIVE
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    return (root / path).resolve()


def legacy_predictor_cache_path(*, repo_root: str | Path) -> Path:
    root = Path(repo_root).resolve()
    return root.parent / "MARCO" / DEFAULT_PREDICTOR_CACHE_RELATIVE


def _shell_export(name: str, value: str) -> str:
    return f"export {name}={shlex.quote(value)}"


def build_predictor_exports(
    *,
    repo_root: str | Path,
    environ: Mapping[str, str] | None = None,
    admet_port: int = DEFAULT_ADMET_PORT,
    drd2_port: int = DEFAULT_DRD2_PORT,
    host: str = DEFAULT_PREDICTOR_HOST,
    include_mock_flag: bool = True,
    mock_mode: str | None = None,
) -> str:
    env = _env_map(environ)
    endpoints = predictor_endpoints(
        environ=env,
        admet_port=admet_port,
        drd2_port=drd2_port,
        host=host,
    )
    cache_path = predictor_cache_path(repo_root=repo_root, environ=env)

    lines = [
        _shell_export("MARCO_ADMET_API", endpoints["MARCO_ADMET_API"]),
        _shell_export("MARCO_DRD2_API", endpoints["MARCO_DRD2_API"]),
        _shell_export("MARCO_PREDICTOR_CACHE_PATH", str(cache_path)),
    ]
    if include_mock_flag:
        value = mock_mode if mock_mode is not None else env.get("MARCO_MOCK_PREDICTOR")
        if value is not None and str(value).strip() != "":
            lines.append(_shell_export("MARCO_MOCK_PREDICTOR", str(value)))
    return "\n".join(lines)


def is_port_in_use(port: int, *, bind_host: str = "0.0.0.0") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((bind_host, int(port)))
        except OSError:
            return True
    return False


def is_tcp_port_open(host: str, port: int, *, timeout_s: float = 1.0) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(float(timeout_s))
        return sock.connect_ex((host, int(port))) == 0


def wait_for_tcp_port(
    host: str,
    port: int,
    *,
    timeout_s: float = 180.0,
    poll_interval_s: float = 1.0,
) -> bool:
    deadline = time.monotonic() + float(timeout_s)
    while True:
        if is_tcp_port_open(host, int(port), timeout_s=min(1.0, poll_interval_s)):
            return True
        now = time.monotonic()
        if now >= deadline:
            return False
        time.sleep(min(float(poll_interval_s), deadline - now))


def _coerce_float(value: object) -> float:
    current = value
    if isinstance(current, list):
        if not current:
            raise ValueError("empty list")
        current = current[0]
    return float(current)


def probe_predictor_endpoint(
    endpoint: str,
    *,
    expected_keys: Sequence[str],
    timeout_s: float = 2.0,
    probe_smiles: str = DEFAULT_PROBE_SMILES,
) -> tuple[bool, str]:
    candidate_keys = [str(item).strip() for item in expected_keys if str(item).strip()]
    if not candidate_keys:
        return False, "no expected keys configured"

    payload = {"smiles": probe_smiles}
    try:
        session = requests.Session()
        session.trust_env = False
        response = session.post(endpoint, json=payload, timeout=float(timeout_s))
    except requests.RequestException as exc:
        return False, f"request failed: {exc}"

    if response.status_code >= 400:
        return False, f"http status {response.status_code}"

    try:
        body = response.json()
    except ValueError:
        return False, "response is not JSON"

    if not isinstance(body, dict):
        return False, f"response JSON type is {type(body).__name__}, expected dict"

    for key in candidate_keys:
        if key not in body:
            continue
        try:
            _coerce_float(body[key])
            return True, f"ok ({key})"
        except (TypeError, ValueError):
            return False, f"key '{key}' exists but value is not numeric"

    return False, f"missing expected keys: {', '.join(candidate_keys)}"


def wait_for_predictor_endpoint(
    endpoint: str,
    *,
    expected_keys: Sequence[str],
    timeout_s: float = 180.0,
    poll_interval_s: float = 1.0,
    request_timeout_s: float = 30.0,
    probe_smiles: str = DEFAULT_PROBE_SMILES,
) -> tuple[bool, str]:
    deadline = time.monotonic() + float(timeout_s)
    last_reason = "not probed"
    while True:
        remaining_s = max(0.1, deadline - time.monotonic())
        ok, reason = probe_predictor_endpoint(
            endpoint,
            expected_keys=expected_keys,
            timeout_s=min(float(request_timeout_s), remaining_s),
            probe_smiles=probe_smiles,
        )
        if ok:
            return True, reason
        last_reason = reason
        now = time.monotonic()
        if now >= deadline:
            return False, last_reason
        time.sleep(min(float(poll_interval_s), deadline - now))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Utilities for MARCO predictor service startup and cache paths."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_exports = subparsers.add_parser("print-exports", help="Print shell exports for predictor env vars.")
    p_exports.add_argument("--repo-root", required=True)
    p_exports.add_argument("--admet-port", type=int, default=DEFAULT_ADMET_PORT)
    p_exports.add_argument("--drd2-port", type=int, default=DEFAULT_DRD2_PORT)
    p_exports.add_argument("--host", default=DEFAULT_PREDICTOR_HOST)
    p_exports.add_argument("--mock-mode", default=None)
    p_exports.add_argument("--exclude-mock-flag", action="store_true")

    p_cache = subparsers.add_parser("cache-path", help="Print resolved predictor cache path.")
    p_cache.add_argument("--repo-root", required=True)

    p_legacy = subparsers.add_parser(
        "legacy-cache-path", help="Print legacy MARCO predictor cache path for migration."
    )
    p_legacy.add_argument("--repo-root", required=True)

    p_in_use = subparsers.add_parser("port-in-use", help="Return success when the port is already in use.")
    p_in_use.add_argument("--port", type=int, required=True)
    p_in_use.add_argument("--bind-host", default="0.0.0.0")

    p_wait = subparsers.add_parser("wait-port", help="Return success when TCP port becomes reachable.")
    p_wait.add_argument("--host", required=True)
    p_wait.add_argument("--port", type=int, required=True)
    p_wait.add_argument("--timeout-s", type=float, default=180.0)
    p_wait.add_argument("--poll-interval-s", type=float, default=1.0)

    p_probe = subparsers.add_parser(
        "probe-endpoint", help="Return success when endpoint behaves like predictor service."
    )
    p_probe.add_argument("--endpoint", required=True)
    p_probe.add_argument("--expected-key", action="append", required=True)
    p_probe.add_argument("--timeout-s", type=float, default=2.0)
    p_probe.add_argument("--probe-smiles", default=DEFAULT_PROBE_SMILES)

    p_wait_probe = subparsers.add_parser(
        "wait-predictor", help="Wait until endpoint passes predictor-specific semantic probe."
    )
    p_wait_probe.add_argument("--endpoint", required=True)
    p_wait_probe.add_argument("--expected-key", action="append", required=True)
    p_wait_probe.add_argument("--timeout-s", type=float, default=180.0)
    p_wait_probe.add_argument("--poll-interval-s", type=float, default=1.0)
    p_wait_probe.add_argument("--request-timeout-s", type=float, default=30.0)
    p_wait_probe.add_argument("--probe-smiles", default=DEFAULT_PROBE_SMILES)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "print-exports":
        print(
            build_predictor_exports(
                repo_root=args.repo_root,
                admet_port=args.admet_port,
                drd2_port=args.drd2_port,
                host=args.host,
                include_mock_flag=not args.exclude_mock_flag,
                mock_mode=args.mock_mode,
            )
        )
        return 0

    if args.command == "cache-path":
        print(predictor_cache_path(repo_root=args.repo_root))
        return 0

    if args.command == "legacy-cache-path":
        print(legacy_predictor_cache_path(repo_root=args.repo_root))
        return 0

    if args.command == "port-in-use":
        return 0 if is_port_in_use(args.port, bind_host=args.bind_host) else 1

    if args.command == "wait-port":
        ok = wait_for_tcp_port(
            host=args.host,
            port=args.port,
            timeout_s=args.timeout_s,
            poll_interval_s=args.poll_interval_s,
        )
        return 0 if ok else 1

    if args.command == "probe-endpoint":
        ok, reason = probe_predictor_endpoint(
            args.endpoint,
            expected_keys=args.expected_key,
            timeout_s=args.timeout_s,
            probe_smiles=args.probe_smiles,
        )
        if not ok:
            print(reason, file=os.sys.stderr)
            return 1
        return 0

    if args.command == "wait-predictor":
        ok, reason = wait_for_predictor_endpoint(
            args.endpoint,
            expected_keys=args.expected_key,
            timeout_s=args.timeout_s,
            poll_interval_s=args.poll_interval_s,
            request_timeout_s=args.request_timeout_s,
            probe_smiles=args.probe_smiles,
        )
        if not ok:
            print(reason, file=os.sys.stderr)
            return 1
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
