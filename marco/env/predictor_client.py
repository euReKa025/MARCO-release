from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Iterable

import requests

from marco.errors import InvalidEvaluationError, PredictorServiceError
from marco.utils import get_env_float, get_env_int


class PredictorClient:
    def __init__(
        self,
        admet_url: str | None = None,
        drd2_url: str | None = None,
        timeout_s: float | None = None,
        max_retries: int | None = None,
    ):
        self.admet_url = admet_url or os.getenv("MARCO_ADMET_API", "http://127.0.0.1:10086/predict/")
        self.drd2_url = drd2_url or os.getenv("MARCO_DRD2_API", "http://127.0.0.1:10087/predict/")
        self.timeout_s = float(
            timeout_s if timeout_s is not None else get_env_float("MARCO_PREDICTOR_TIMEOUT_S", 60.0)
        )
        self.max_retries = int(
            max_retries if max_retries is not None else get_env_int("MARCO_PREDICTOR_MAX_RETRIES", 0)
        )
        self.mock_mode = os.getenv("MARCO_MOCK_PREDICTOR", "0") == "1"
        self.cache_path = str(os.getenv("MARCO_PREDICTOR_CACHE_PATH", "")).strip()
        self.cache_enabled = bool(self.cache_path)
        self.session = requests.Session()
        self.session.trust_env = False
        self._memory_cache: dict[tuple[str, str], dict] = {}
        if self.cache_enabled:
            self._init_cache()

    def predict(self, smiles: str, required_properties: Iterable[str]) -> dict[str, float]:
        required = sorted(set(required_properties))
        if self.mock_mode:
            return self._mock_predict(smiles, required)

        need_drd2 = "drd2" in required
        admet_required = [prop for prop in required if prop != "drd2"]
        admet_payload = self._predict_service(
            "admet",
            self.admet_url,
            smiles,
            required_keys=admet_required,
        )
        if not isinstance(admet_payload, dict):
            raise PredictorServiceError("ADMET response is not a dict")

        merged = dict(admet_payload)
        if need_drd2:
            drd2_payload = self._predict_service("drd2", self.drd2_url, smiles, required_keys=["drd2"])
            if not isinstance(drd2_payload, dict):
                raise PredictorServiceError("DRD2 response is not a dict")
            merged.update(drd2_payload)

        output: dict[str, float] = {}
        for prop in required:
            if prop not in merged:
                raise InvalidEvaluationError(f"Missing property '{prop}' in predictor output")
            val = self._as_float(merged[prop], prop)
            output[prop] = val
        return output

    def _post_with_retry(self, url: str, smiles: str):
        payload = {"smiles": smiles}
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.post(url, json=payload, timeout=self.timeout_s)
                response.raise_for_status()
                return response.json()
            except requests.RequestException as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(0.2 * (attempt + 1))
                    continue
                break
        raise PredictorServiceError(f"Predictor request failed: {last_error}")

    def _predict_service(self, service: str, url: str, smiles: str, *, required_keys: Iterable[str] = ()):
        cached = self._cache_get(service, smiles)
        required = [str(key) for key in required_keys if str(key)]
        if cached is not None and all(key in cached for key in required):
            return dict(cached)

        payload = self._post_with_retry(url, smiles)
        if isinstance(payload, dict):
            self._cache_set(service, smiles, payload)
        return payload

    def _init_cache(self) -> None:
        try:
            cache_path = Path(self.cache_path)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(str(cache_path), timeout=30.0) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS predictor_cache (
                        service TEXT NOT NULL,
                        smiles TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        updated_at REAL NOT NULL,
                        PRIMARY KEY (service, smiles)
                    )
                    """
                )
        except sqlite3.Error:
            self.cache_enabled = False

    def _cache_get(self, service: str, smiles: str) -> dict | None:
        if not self.cache_enabled:
            return None

        key = (service, smiles)
        cached = self._memory_cache.get(key)
        if cached is not None:
            return dict(cached)

        try:
            with sqlite3.connect(self.cache_path, timeout=30.0) as conn:
                row = conn.execute(
                    "SELECT payload_json FROM predictor_cache WHERE service = ? AND smiles = ?",
                    (service, smiles),
                ).fetchone()
        except sqlite3.Error:
            return None

        if row is None:
            return None

        try:
            payload = json.loads(str(row[0]))
        except json.JSONDecodeError:
            return None

        if not isinstance(payload, dict):
            return None

        self._memory_cache[key] = dict(payload)
        return dict(payload)

    def _cache_set(self, service: str, smiles: str, payload: dict) -> None:
        if not self.cache_enabled:
            return

        key = (service, smiles)
        payload_copy = dict(payload)
        self._memory_cache[key] = payload_copy
        try:
            with sqlite3.connect(self.cache_path, timeout=30.0) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute(
                    """
                    INSERT OR REPLACE INTO predictor_cache(service, smiles, payload_json, updated_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (service, smiles, json.dumps(payload_copy, sort_keys=True), time.time()),
                )
                conn.commit()
        except sqlite3.Error:
            return

    @staticmethod
    def _as_float(value, prop: str) -> float:
        try:
            if isinstance(value, list):
                if not value:
                    raise ValueError("empty list")
                value = value[0]
            return float(value)
        except (TypeError, ValueError) as exc:
            raise InvalidEvaluationError(f"Property '{prop}' cannot be converted to float: {value}") from exc

    @staticmethod
    def _mock_predict(smiles: str, required: list[str]) -> dict[str, float]:
        key = hashlib.sha256(smiles.encode("utf-8")).hexdigest()
        seed_val = int(key[:8], 16)

        def gen(scale: float, shift: float = 0.0) -> float:
            return ((seed_val % 10000) / 10000.0) * scale + shift

        base = {
            "bbbp": gen(1.0),
            "drd2": gen(1.0, 0.05),
            "qed": gen(1.0),
            "plogp": gen(6.0, -3.0),
            "mutagenicity": gen(1.0),
            "hia": gen(1.0),
        }
        return {k: float(base[k]) for k in required}
