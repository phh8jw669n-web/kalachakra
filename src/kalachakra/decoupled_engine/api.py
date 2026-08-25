"""FastAPI server bridging the decoupled engine to the web dashboard / WebGL.

Endpoints
---------
GET  /                           -> the live global energy-signature dashboard
GET  /health                     -> model + timeline-coverage summary
GET  /api/coastlines.geojson     -> world outline for the map overlay
GET  /api/texture                -> raw gamma-sRGB byte buffer (energy layer) for a
                                    timestamp, with X-Width / X-Height / X-JD headers
POST /api/point                  -> {oklab, rgb, attribution} for one (lat, lon)
POST /api/similar                -> nearest historical snapshots by tension-vector
                                    cosine similarity (latent timeline search)

If no trained checkpoint is supplied the server starts in *demo* mode with a
randomly-initialised model, so the dashboard is fully explorable before training
finishes (the field is then noise, not a learned signature).

A small in-memory latent bank of historical tension vectors is built at startup for
the similarity search; snapshots the active ephemeris backend cannot cover are
skipped gracefully.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..ephemeris.calendar import format_jd
from .bundle import build_models
from .config import EngineConfig
from .inference import DecoupledInference, jd_from_timestamp

_ROOT = Path(__file__).resolve().parents[3]
_WEB = _ROOT / "web" / "decoupled.html"
_COAST = _ROOT / "web" / "coastlines.geojson"


def _load_or_demo(ckpt_path, device, ephe_path, jpl_file):
    """Return ``(engine, is_demo)`` -- a loaded checkpoint or a random-init demo."""
    from ..ephemeris import global_state as gs
    if ckpt_path and Path(ckpt_path).exists():
        return DecoupledInference.from_checkpoint(
            ckpt_path, device=device, ephe_path=ephe_path, jpl_file=jpl_file), False
    if gs.ephemeris_available():
        gs.configure_from_args(ephe_path=ephe_path, jpl_file=jpl_file)
    cfg = EngineConfig()
    # A demo has no trained span; default the dashboard to a modern window the
    # default (Moshier) backend fully covers, so it works with no data files.
    from ..ephemeris.calendar import gregorian_to_jd
    cfg.data.start_jd = gregorian_to_jd(1950, 1, 1)
    cfg.data.end_jd = gregorian_to_jd(2050, 1, 1)
    sky, earth = build_models(cfg)
    return DecoupledInference(sky, earth, cfg, device=device), True


def _build_latent_bank(engine: DecoupledInference, n: int):
    """Sample ``n`` snapshot tension vectors across the configured timeline span."""
    cfg = engine.cfg.data
    jds = np.linspace(cfg.start_jd, cfg.end_jd, n)
    ok_jds, vecs = [], []
    for jd in jds:
        try:
            v = engine.tension_vector(float(jd))[0].float().cpu().numpy()
        except Exception:  # noqa: BLE001 -- backend may not cover deep time
            continue
        ok_jds.append(float(jd))
        vecs.append(v)
    if not vecs:
        return [], np.zeros((0, engine.cfg.sky.tension_dim), dtype=np.float32)
    mat = np.stack(vecs, axis=0)
    mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
    return ok_jds, mat.astype(np.float32)


def _coverage(engine: DecoupledInference) -> dict:
    """The timeline span the dashboard's time slider should default to."""
    d = engine.cfg.data
    return {"start_jd": float(d.start_jd), "end_jd": float(d.end_jd),
            "start": format_jd(d.start_jd), "end": format_jd(d.end_jd)}


def create_app(ckpt_path: str | None = None, device: str = "", bank_size: int = 64,
               ephe_path: str | None = None, jpl_file: str | None = None):
    from fastapi import Body, FastAPI, Response
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, JSONResponse

    engine, is_demo = _load_or_demo(ckpt_path, device, ephe_path, jpl_file)
    bank_jds, bank_vecs = _build_latent_bank(engine, bank_size)

    app = FastAPI(title="Decoupled Projection Engine", version="1.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                       allow_headers=["*"])

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "device": str(engine.device), "demo": is_demo,
                "tension_dim": engine.cfg.sky.tension_dim,
                "bodies": list(engine.sky.body_names()),
                "coverage": _coverage(engine), "latent_bank": len(bank_jds)}

    @app.get("/api/coastlines.geojson")
    def coastlines():
        if _COAST.is_file():
            return FileResponse(str(_COAST), media_type="application/geo+json")
        return Response("{}", media_type="application/json")

    @app.get("/api/texture")
    def texture(timestamp: str, width: int = 512, height: int = 256):
        try:
            tex = engine.global_texture(timestamp, width=width, height=height)
        except Exception as exc:  # noqa: BLE001 -- e.g. timestamp outside ephemeris
            return JSONResponse({"error": f"{exc}"}, status_code=400)
        return Response(content=tex["bytes"], media_type="application/octet-stream",
                        headers={"X-Width": str(width), "X-Height": str(height),
                                 "X-Channels": "3", "X-JD": str(tex["jd"])})

    @app.post("/api/point")
    def point(body: dict = Body(...)) -> JSONResponse:
        try:
            r = engine.pinpoint(body["timestamp"], float(body["lat"]),
                                float(body["lon"]))
        except Exception as exc:  # noqa: BLE001 -- bad args or out-of-range timestamp
            return JSONResponse({"error": f"{exc}"}, status_code=400)
        return JSONResponse({
            "jd": r["jd"], "date": format_jd(r["jd"]), "lat": r["lat"], "lon": r["lon"],
            "oklab": r["oklab"].tolist(), "rgb": r["rgb8"].tolist(),
            "attribution": r["attribution"],
        })

    @app.post("/api/similar")
    def similar(body: dict = Body(...)) -> JSONResponse:
        if not bank_jds:
            return JSONResponse({"error": "empty latent bank"}, status_code=503)
        try:
            jd = jd_from_timestamp(body["timestamp"])
            k = int(body.get("k", 5))
        except (KeyError, ValueError) as exc:
            return JSONResponse({"error": f"bad request: {exc}"}, status_code=400)
        q = engine.tension_vector(jd)[0].float().cpu().numpy()
        q = q / (np.linalg.norm(q) + 1e-9)
        sims = bank_vecs @ q
        order = np.argsort(-sims)[:k]
        return JSONResponse({"query_jd": jd, "matches": [
            {"jd": bank_jds[i], "date": format_jd(bank_jds[i]),
             "similarity": float(sims[i])} for i in order]})

    @app.get("/")
    def index():
        if _WEB.is_file():
            return FileResponse(str(_WEB))
        return Response("web/decoupled.html not found", status_code=404)

    return app


def serve(ckpt_path: str | None = None, host: str = "127.0.0.1", port: int = 8100,
          device: str = "", **kw) -> int:
    try:
        import uvicorn
    except ImportError:
        print('ERROR: uvicorn not installed. `pip install "kalachakra[serve]"`.')
        return 2
    app = create_app(ckpt_path, device=device, **kw)
    tag = "DEMO (random weights)" if not (ckpt_path and Path(ckpt_path).exists()) \
        else ckpt_path
    print(f"\nDecoupled engine dashboard on http://{host}:{port}  [{tag}]")
    uvicorn.run(app, host=host, port=port)
    return 0
