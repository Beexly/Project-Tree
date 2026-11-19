"""Tests for Knowledge Integrity API X-API-Key auth and JSONL usage metering."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from server.app import app  # noqa: E402

SAMPLE_GRAPH = {
    "nodes": [{"id": "a", "features": [1.0, 0.0]}, {"id": "b", "features": [1.0, 0.0]}],
    "edges": [{"source": "a", "target": "b", "relation": "same", "theta": 0.0}],
    "feature_dim": 2,
}


def test_health_open_even_when_api_keys_set(tmp_path, monkeypatch):
    log_file = tmp_path / "usage.jsonl"
    monkeypatch.setenv("CRC_API_KEYS", "key-alpha,key-beta")
    monkeypatch.setenv("CRC_USAGE_LOG", str(log_file))

    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_missing_or_bad_api_key_returns_401(tmp_path, monkeypatch):
    log_file = tmp_path / "usage.jsonl"
    monkeypatch.setenv("CRC_API_KEYS", "key-alpha,key-beta")
    monkeypatch.setenv("CRC_USAGE_LOG", str(log_file))

    client = TestClient(app)

    r_missing = client.post("/v1/consistency", json=SAMPLE_GRAPH)
    assert r_missing.status_code == 401

    r_bad = client.post(
        "/v1/consistency",
        json=SAMPLE_GRAPH,
        headers={"X-API-Key": "wrong-key"},
    )
    assert r_bad.status_code == 401


def test_valid_api_key_returns_200_and_logs_usage(tmp_path, monkeypatch):
    log_file = tmp_path / "usage.jsonl"
    monkeypatch.setenv("CRC_API_KEYS", "key-alpha,key-beta")
    monkeypatch.setenv("CRC_USAGE_LOG", str(log_file))

    client = TestClient(app)
    resp = client.post(
        "/v1/consistency",
        json=SAMPLE_GRAPH,
        headers={"X-API-Key": "key-beta"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["repair_class"] == "consistent"

    lines = [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1
    assert lines[0]["path"] == "/v1/consistency"
    assert lines[0]["status_code"] == 200
    assert lines[0]["authorized"] is True
    assert lines[0]["key_prefix"] == "key-be"
