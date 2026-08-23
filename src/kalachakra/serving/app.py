"""
FastAPI control plane + binary WebSocket stream (blueprint §7).

Strict separation of concerns:
  * lightweight JSON/REST for control + metadata (Pydantic-validated),
  * dense field data over a binary WebSocket (see :mod:`kalachakra.serving.binary`).

The REST ``/inspect`` endpoint routes a viewport query through the DuckDB engine
(tier selection, partition pruning, H3 filtering, rarity threshold) and returns
metadata + news cards; ``/stream`` pushes packed field frames for playback.

Requires fastapi (and an ASGI server such as uvicorn to run).
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:  # pragma: no cover - optional dependency
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from pydantic import BaseModel, Field

    _HAS_FASTAPI = True
except Exception:  # noqa: BLE001
    _HAS_FASTAPI = False

from ..analysis import radar
from ..analysis.rarity import RarityModel
from ..ephemeris.calendar import parse_datetime
from . import binary
from ..storage.duckdb_engine import DuckDBEngine, ViewportQuery
from ..storage.parquet_store import ParquetTokenStore


def fastapi_available() -> bool:
    return _HAS_FASTAPI


def _require():
    if not _HAS_FASTAPI:
        raise RuntimeError("fastapi is required. `pip install \"kalachakra[serve]\"`.")


# Pydantic models must live at module scope: with `from __future__ import
# annotations` in effect, FastAPI resolves the string annotations via module
# globals, so locally-scoped models fail to resolve (treated as query params).
if _HAS_FASTAPI:
    class InspectRequest(BaseModel):
        min_lat: float = Field(-90, ge=-90, le=90)
        min_lng: float = Field(-180, ge=-180, le=180)
        max_lat: float = Field(90, ge=-90, le=90)
        max_lng: float = Field(180, ge=-180, le=180)
        start: str                       # ISO datetime (UTC)
        end: str
        velocity: float = 1.0
        rarity_min: float = 0.0
        limit: int = Field(1000, ge=1, le=10000)

    class InspectResponse(BaseModel):
        tier: str
        span_years: float
        significance_percentile: float
        band_gains: dict[str, float]
        n_rows: int
        rows: list[dict[str, Any]]


def create_app(store_root: str):
    """Build the FastAPI app over a Parquet token store at ``store_root``."""
    _require()

    app = FastAPI(title="Kalachakra Cosmic Weather Radar", version="0.2.0")
    store = ParquetTokenStore(store_root)
    engine = DuckDBEngine(store)

    def _query(req: "InspectRequest"):
        j0 = parse_datetime(req.start)
        j1 = parse_datetime(req.end)
        q = ViewportQuery(req.min_lat, req.min_lng, req.max_lat, req.max_lng,
                          j0, j1, velocity=req.velocity,
                          rarity_min=req.rarity_min, limit=req.limit)
        return q, engine.query(q)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok",
                "tier1": store.has_tier("tier1"),
                "tier2": store.has_tier("tier2")}

    @app.post("/inspect", response_model=InspectResponse)
    def inspect(req: InspectRequest) -> "InspectResponse":
        q, rows = _query(req)
        span_years = max((q.end_jd - q.start_jd) / 365.25, 1e-6)
        tier = engine._tier_for(q)
        # Trim heavy latent vectors out of the JSON payload (they stream binary).
        light = [{k: v for k, v in r.items() if k != "latent"} for r in rows]
        return InspectResponse(
            tier=tier, span_years=span_years,
            significance_percentile=radar.significance_percentile(span_years),
            band_gains=radar.band_gains(radar.temporal_stride(
                int((q.end_jd - q.start_jd) * 86400 / 24))),
            n_rows=len(rows), rows=light,
        )

    @app.websocket("/stream")
    async def stream(ws: WebSocket) -> None:
        await ws.accept()
        try:
            while True:
                req = InspectRequest(**(await ws.receive_json()))
                _q, rows = _query(req)
                if not rows:
                    await ws.send_bytes(binary.pack_frame(
                        *[np.zeros(0, np.float32)] * 4,
                        np.zeros(0, np.uint16), np.zeros(0, np.uint16)))
                    continue
                cols = {k: np.array([r[k] for r in rows]) for k in
                        ("lat", "lng", "potential", "shear", "macro", "micro")}
                await ws.send_bytes(binary.pack_frame(
                    cols["lat"].astype(np.float32), cols["lng"].astype(np.float32),
                    cols["potential"].astype(np.float32),
                    cols["shear"].astype(np.float32),
                    cols["macro"].astype(np.uint16), cols["micro"].astype(np.uint16)))
        except WebSocketDisconnect:
            return

    app.state.engine = engine
    return app
