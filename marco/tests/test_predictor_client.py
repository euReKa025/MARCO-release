from __future__ import annotations

from pathlib import Path

import pytest

from marco.env.predictor_client import PredictorClient


def _patch_predictor_session(monkeypatch, fake_post, sessions: list | None = None):  # noqa: ANN001
    class _FakeSession:
        def __init__(self):
            self.trust_env = True
            if sessions is not None:
                sessions.append(self)

        def post(self, url, json=None, timeout=None):  # noqa: ANN001
            return fake_post(url, json=json, timeout=timeout)

    monkeypatch.setattr("marco.env.predictor_client.requests.Session", _FakeSession)


def test_predictor_client_reads_timeout_and_retry_from_env(monkeypatch):
    monkeypatch.setenv("MARCO_PREDICTOR_TIMEOUT_S", "60.0")
    monkeypatch.setenv("MARCO_PREDICTOR_MAX_RETRIES", "0")

    client = PredictorClient()

    assert client.timeout_s == pytest.approx(60.0)
    assert client.max_retries == 0


def test_predictor_client_reuses_sqlite_cache_across_calls_and_instances(monkeypatch, tmp_path: Path):
    cache_path = tmp_path / "predictor_cache.sqlite3"
    monkeypatch.setenv("MARCO_PREDICTOR_CACHE_PATH", str(cache_path))

    calls: list[str] = []

    class _Response:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def fake_post(url, json=None, timeout=None):  # noqa: ANN001
        calls.append(url)
        smiles = json["smiles"]
        if "10086" in url:
            return _Response({"bbbp": 0.6, "plogp": 1.2, "echo": smiles})
        return _Response({"drd2": 0.2})

    _patch_predictor_session(monkeypatch, fake_post)

    client = PredictorClient()
    out1 = client.predict("CCO", ["bbbp", "drd2"])
    assert out1 == {"bbbp": pytest.approx(0.6), "drd2": pytest.approx(0.2)}
    assert len(calls) == 2

    out2 = client.predict("CCO", ["bbbp", "drd2"])
    assert out2 == out1
    assert len(calls) == 2

    client2 = PredictorClient()
    out3 = client2.predict("CCO", ["bbbp", "drd2"])
    assert out3 == out1
    assert len(calls) == 2

    assert cache_path.is_file()


def test_predictor_client_without_drd2_only_uses_admet_cache(monkeypatch, tmp_path: Path):
    cache_path = tmp_path / "predictor_cache.sqlite3"
    monkeypatch.setenv("MARCO_PREDICTOR_CACHE_PATH", str(cache_path))

    calls: list[str] = []

    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"bbbp": 0.4, "plogp": -0.1}

    def fake_post(url, json=None, timeout=None):  # noqa: ANN001
        calls.append(url)
        return _Response()

    _patch_predictor_session(monkeypatch, fake_post)

    client = PredictorClient()
    out1 = client.predict("CCN", ["bbbp"])
    out2 = client.predict("CCN", ["bbbp"])

    assert out1 == {"bbbp": pytest.approx(0.4)}
    assert out2 == out1
    assert calls == ["http://127.0.0.1:10086/predict/"]


def test_predictor_client_refreshes_stale_admet_cache_missing_required_key(monkeypatch, tmp_path: Path):
    cache_path = tmp_path / "predictor_cache.sqlite3"
    monkeypatch.setenv("MARCO_PREDICTOR_CACHE_PATH", str(cache_path))

    calls: list[str] = []

    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"bbbp": 0.4, "hia": 0.8, "plogp": -0.1}

    def fake_post(url, json=None, timeout=None):  # noqa: ANN001
        calls.append(url)
        return _Response()

    _patch_predictor_session(monkeypatch, fake_post)

    old_client = PredictorClient()
    old_client._cache_set("admet", "CCN", {"bbbp": 0.4, "plogp": -0.1})

    client = PredictorClient()
    out = client.predict("CCN", ["bbbp", "hia"])

    assert out == {"bbbp": pytest.approx(0.4), "hia": pytest.approx(0.8)}
    assert calls == ["http://127.0.0.1:10086/predict/"]

    client2 = PredictorClient()
    out2 = client2.predict("CCN", ["hia"])
    assert out2 == {"hia": pytest.approx(0.8)}
    assert calls == ["http://127.0.0.1:10086/predict/"]


def test_predictor_client_disables_proxy_env_for_local_predictors(monkeypatch):
    sessions = []

    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"bbbp": 0.4}

    def fake_post(url, json=None, timeout=None):  # noqa: ANN001
        return _Response()

    _patch_predictor_session(monkeypatch, fake_post, sessions=sessions)

    client = PredictorClient()
    assert client.predict("CCN", ["bbbp"]) == {"bbbp": pytest.approx(0.4)}
    assert sessions
    assert sessions[0].trust_env is False
