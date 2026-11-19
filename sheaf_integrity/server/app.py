"""
FastAPI surface for Knowledge Integrity API v0.2.
Run: uvicorn server.app:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core"))

from crc import EdgeMap, crc_connection, crc_constant_graph  # noqa: E402

app = FastAPI(
    title="Knowledge Integrity API",
    version="0.2.0",
    description="CRC consistency + proposals. No Riemannian jargon on the wire.",
)


def _allowed_api_keys() -> set[str]:
    raw = os.environ.get("CRC_API_KEYS", "")
    return {k.strip() for k in raw.split(",") if k.strip()}


def _usage_log_path() -> Path:
    return Path(os.environ.get("CRC_USAGE_LOG", "/tmp/crc_usage.jsonl"))


def _append_usage_log(entry: dict[str, Any]) -> None:
    log_path = _usage_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


@app.middleware("http")
async def api_key_and_metering_middleware(request: Request, call_next):
    t0 = time.perf_counter()
    path = request.url.path
    keys = _allowed_api_keys()
    provided_key = request.headers.get("X-API-Key", "")

    if keys and path != "/health":
        if not provided_key or provided_key not in keys:
            resp = JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing X-API-Key"},
            )
            _append_usage_log({
                "ts": datetime.now(timezone.utc).isoformat(),
                "method": request.method,
                "path": path,
                "status_code": 401,
                "authorized": False,
                "duration_ms": round((time.perf_counter() - t0) * 1000.0, 3),
            })
            return resp

    response = await call_next(request)
    _append_usage_log({
        "ts": datetime.now(timezone.utc).isoformat(),
        "method": request.method,
        "path": path,
        "status_code": response.status_code,
        "authorized": True,
        "key_prefix": provided_key[:6] if provided_key else None,
        "duration_ms": round((time.perf_counter() - t0) * 1000.0, 3),
    })
    return response


class NodeIn(BaseModel):
    id: str
    features: list[float] | None = None
    system: str | None = None


class EdgeIn(BaseModel):
    source: str
    target: str
    relation: str = "related"
    theta: float = 0.0


class GraphIn(BaseModel):
    nodes: list[NodeIn]
    edges: list[EdgeIn]
    feature_dim: int = 2
    reference_features: dict[str, list[float]] | None = None
    top_k: int = 5


def _build(payload: GraphIn):
    nodes = [n.id for n in payload.nodes]
    d = payload.feature_dim
    features: dict[str, np.ndarray] = {}
    for n in payload.nodes:
        if n.features is None:
            features[n.id] = np.zeros(d)
        else:
            v = np.asarray(n.features, dtype=float).ravel()
            if v.size < d:
                v = np.pad(v, (0, d - v.size))
            features[n.id] = v[:d]

    use_connection = d == 2 and any(abs(e.theta) > 1e-12 for e in payload.edges)
    reference = None
    if payload.reference_features:
        reference = {
            k: np.asarray(v, dtype=float).ravel()[:d]
            for k, v in payload.reference_features.items()
        }

    if use_connection or d == 2:
        maps = [
            EdgeMap(e.source, e.target, e.relation, float(e.theta)) for e in payload.edges
        ]
        feats2 = {k: v[:2] if v.size >= 2 else np.pad(v, (0, 2 - v.size)) for k, v in features.items()}
        ref2 = None
        if reference:
            ref2 = {
                k: v[:2] if v.size >= 2 else np.pad(v, (0, max(0, 2 - v.size)))
                for k, v in reference.items()
            }
        report = crc_connection(nodes, maps, feats2, reference=ref2)
    else:
        edges = [(e.source, e.target, e.relation) for e in payload.edges]
        report = crc_constant_graph(nodes, edges, features, d, reference=reference)
    return report


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "knowledge-integrity", "version": "0.2.0"}


@app.post("/v1/consistency")
def consistency(body: GraphIn) -> dict[str, Any]:
    try:
        report = _build(body)
        return report.to_dict()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/propose")
def propose(body: GraphIn) -> dict[str, Any]:
    try:
        report = _build(body)
        props = report.proposals[: body.top_k]
        return {
            "baseline_energy": report.energy,
            "repair_class": report.repair_class,
            "proposals": props,
        }
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
