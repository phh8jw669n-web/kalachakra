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

import numpy as _np

from .. import constants as C
from ..analysis import radar, weather
from ..ephemeris import bodies, global_state
from ..ephemeris.calendar import format_jd, parse_datetime
from . import binary
from ..projection import spatial
from ..projection.microgrid import bbox_microgrid, resolution_km
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
        cluster_min_size: int = Field(0, ge=0)   # 0 -> auto from row count

    class InspectResponse(BaseModel):
        tier: str
        span_years: float
        significance_percentile: float
        band_gains: dict[str, float]
        n_rows: int
        rows: list[dict[str, Any]]
        global_latent: list[float] | None = None   # mean latent -> SH topography
        n_clusters: int | None = None              # HDBSCAN manifold clusters
        cluster_method: str | None = None          # "hdbscan" | "fallback"

    class NewsRequest(BaseModel):
        min_lat: float = Field(-90, ge=-90, le=90)
        min_lng: float = Field(-180, ge=-180, le=180)
        max_lat: float = Field(90, ge=-90, le=90)
        max_lng: float = Field(180, ge=-180, le=180)
        start: str
        end: str
        rarity_min: float = 0.0
        top: int = Field(12, ge=1, le=100)

    class MicrogridRequest(BaseModel):
        min_lat: float = Field(..., ge=-90, le=90)
        min_lng: float = Field(..., ge=-180, le=180)
        max_lat: float = Field(..., ge=-90, le=90)
        max_lng: float = Field(..., ge=-180, le=180)
        datetime: str
        density: int = Field(48, ge=2, le=256)

    class TelemetryRequest(BaseModel):
        lat: float = Field(..., ge=-90, le=90)
        lng: float = Field(..., ge=-180, le=180)
        datetime: str


def _cluster_rows(rows, latents, cluster_min_size):
    """Cluster the returned latents (§6.2) and stamp ``cluster_id`` on each row.

    Rows without a latent (e.g. tier-2/3 rollups) get ``cluster_id = -1`` (noise).
    Returns ``(n_clusters, method)`` or ``(None, None)`` when there is nothing to
    cluster.
    """
    for r in rows:
        r["cluster_id"] = -1
    if len(latents) < 2:
        return None, None
    from ..analysis.clustering import cluster_latents
    mcs = cluster_min_size or max(2, len(latents) // 20)
    res = cluster_latents(_np.asarray(latents, dtype=float), min_cluster_size=mcs)
    li = 0
    for r in rows:
        if r.get("latent") is not None:
            r["cluster_id"] = int(res.labels[li])
            li += 1
    return res.n_clusters, res.method


def _applying_flag(lons, speeds, sig):
    if not sig.dominant_aspects:
        return None
    top = sig.dominant_aspects[0]
    ai = bodies.index_of(top["bodies"][0])
    bi = bodies.index_of(top["bodies"][1])
    return radar.is_applying(lons, speeds, ai, bi)


def _lookup_stored(engine, store, jd: float, lat: float, lng: float) -> dict:
    """Best-effort stored token/rarity for a coordinate+time from the index."""
    if not store.has_tier("tier1"):
        return {}
    dt = 1.0 / 24.0  # +/- 1 hour window
    q = ViewportQuery(lat - 1.0, lng - 1.0, lat + 1.0, lng + 1.0,
                      jd - dt, jd + dt, rarity_min=0.0, limit=1)
    rows = engine.query(q)
    if not rows:
        return {}
    r = rows[0]
    return {"rarity_percentile": round(float(r.get("rarity", 0.0)) * 100.0, 3),
            "macro": int(r.get("macro", 0)), "micro": int(r.get("micro", 0))}


def create_app(store_root: str):
    """Build the FastAPI app over a Parquet token store at ``store_root``."""
    _require()

    app = FastAPI(title="Kalachakra Cosmic Weather Radar", version="0.3.0")
    from .webui import enable_cors
    enable_cors(app)
    store = ParquetTokenStore(store_root)
    engine = DuckDBEngine(store)
    # Honor a saved full-span backend so live micro-grid / telemetry queries reach
    # the far past/future; falls back to Moshier otherwise.
    if global_state.ephemeris_available():
        global_state.auto_configure()

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
                "tier2": store.has_tier("tier2"),
                "tier3": store.has_tier("tier3"),
                "projection_version": store.projection_version(),
                "current_projection_version": C.PROJECTION_VERSION}

    @app.post("/inspect", response_model=InspectResponse)
    def inspect(req: InspectRequest) -> "InspectResponse":
        q, rows = _query(req)
        span_years = max((q.end_jd - q.start_jd) / 365.25, 1e-6)
        tier = engine._tier_for(q)
        # Mean latent -> global SH topography coefficients for the client.
        latents = [r["latent"] for r in rows if r.get("latent") is not None]
        global_latent = (_np.mean(_np.asarray(latents, dtype=float), axis=0).tolist()
                         if latents else None)
        # §6.2 density-based manifold clustering (HDBSCAN, or a dependency-free
        # fallback) over the returned latents; attach a cluster id per row.
        n_clusters, cluster_method = _cluster_rows(rows, latents, req.cluster_min_size)
        # Trim heavy latent vectors out of the JSON payload (they stream binary).
        light = [{k: v for k, v in r.items() if k != "latent"} for r in rows]
        return InspectResponse(
            tier=tier, span_years=span_years,
            significance_percentile=radar.significance_percentile(span_years),
            band_gains=radar.band_gains(radar.temporal_stride(
                int((q.end_jd - q.start_jd) * 86400 / 24))),
            n_rows=len(rows), rows=light, global_latent=global_latent,
            n_clusters=n_clusters, cluster_method=cluster_method,
        )

    @app.post("/news")
    def news(req: NewsRequest) -> dict:
        """§6 textless geometric news radar: the rarest events in a viewport as
        pure-geometry news cards (body Cartesian vectors, applying/separating,
        dominant aspect, rarity percentile — no interpreted text)."""
        if not global_state.ephemeris_available():
            return {"error": "pyswisseph not installed"}
        j0, j1 = parse_datetime(req.start), parse_datetime(req.end)
        q = ViewportQuery(req.min_lat, req.min_lng, req.max_lat, req.max_lng,
                          j0, j1, rarity_min=req.rarity_min, limit=req.top)
        rows = engine.query(q)                      # already ORDER BY rarity DESC
        cards = []
        for r in rows[:req.top]:
            jd = float(r["jd"])
            g = global_state.global_state_frame(jd)
            rarity = float(r.get("rarity", 0.0))
            card = radar.build_news_card(
                jd, float(r["lat"]), float(r["lng"]), g,
                int(r.get("macro", 0)), int(r.get("micro", 0)),
                rarity, round(rarity * 100.0, 3))
            cards.append(card.to_dict())
        return {"n": len(cards), "span_years": max((j1 - j0) / 365.25, 1e-6),
                "significance_percentile": radar.significance_percentile(
                    max((j1 - j0) / 365.25, 1e-6)),
                "cards": cards}

    @app.post("/microgrid")
    def microgrid(req: MicrogridRequest) -> dict:
        """§4 dynamic LOD: compute the real weather field over an on-the-fly
        regional micro-grid via continuous analytical projection (no static mesh)."""
        if not global_state.ephemeris_available():
            return {"error": "pyswisseph not installed"}
        jd = parse_datetime(req.datetime)
        grid = bbox_microgrid(req.min_lat, req.min_lng, req.max_lat, req.max_lng,
                              req.density)
        wm = weather.weather_map(jd, grid)
        sig = wm["signature"]
        return {
            "timestamp": format_jd(jd),
            "density": req.density,
            "n_nodes": grid.n_nodes,
            "resolution_km": resolution_km(req.min_lat, req.min_lng,
                                           req.max_lat, req.max_lng, req.density),
            "lat": _np.rad2deg(grid.lat).round(5).tolist(),
            "lng": _np.rad2deg(grid.lon).round(5).tolist(),
            "potential": wm["potential"].round(5).tolist(),
            "shear": wm["shear"].round(5).tolist(),
            "summary": {"resonance": sig.resonance, "tension": sig.tension,
                        "potential_R": sig.potential,
                        "eclipse": sig.eclipse["is_eclipse"]},
        }

    @app.post("/telemetry")
    def telemetry(req: TelemetryRequest) -> dict:
        """§5 Sidebar Inspector: raw numerical constituents at a coordinate."""
        if not global_state.ephemeris_available():
            return {"error": "pyswisseph not installed"}
        jd = parse_datetime(req.datetime)
        g = global_state.global_state_frame(jd)
        lons = _np.arctan2(g[:, 1], g[:, 0])
        af = weather.aspect_field(lons)
        sig = weather.frame_signature(jd)

        # Local intensity (diurnal / Micro-band term) at the coordinate.
        one = bbox_microgrid(req.lat - 0.01, req.lng - 0.01,
                             req.lat + 0.01, req.lng + 0.01, 2)
        field = spatial.project(g, jd, one)
        local = float(weather.local_intensity(field, af["activation"]).mean())

        entities = []
        for i, name in enumerate(bodies.NAMES):
            if weather.BODY_WEIGHTS[i] == 0:
                continue
            entities.append({
                "name": name,
                "unit_vector": g[i, :3].astype(float).round(6).tolist(),
                "radial_distance_au": float(g[i, 5]),
                "angular_velocity_deg_per_day": float(_np.rad2deg(g[i, 3])),
                "retrograde": bool(g[i, 3] < 0),
            })

        # Stored token/rarity for this coordinate+time, if the index has it.
        stored = _lookup_stored(engine, store, jd, req.lat, req.lng)
        return {
            "timestamp": format_jd(jd), "julian_day": jd,
            "lat": req.lat, "lng": req.lng,
            "rarity_percentile": stored.get("rarity_percentile"),
            "archetype": {"macro": stored.get("macro"), "micro": stored.get("micro")},
            "band_energies": radar.band_energies(af["activation"], diurnal=local),
            "resonance": sig.resonance, "tension": sig.tension,
            "geometric_potential_R": sig.potential,
            "eclipse": sig.eclipse,
            "dominant_aspect": sig.dominant_aspects[0] if sig.dominant_aspects else None,
            "applying": _applying_flag(lons, g[:, 3], sig),
            "entities": entities,
        }

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
