"""FastAPI server bridging the decoupled engine to a React / WebGL frontend.

Endpoints
---------
GET  /health                     -> model + coverage summary
GET  /api/texture                -> raw gamma-sRGB byte buffer (WebGL texture) for a
                                    timestamp, with X-Width / X-Height headers
POST /api/point                  -> {oklab, rgb, attribution} for one (lat, lon)
POST /api/similar                -> nearest historical snapshots by tension-vector
                                    cosine similarity (latent timeline search)

A small in-memory latent bank of historical tension vectors is built at startup for
the similarity search; snapshots the active ephemeris backend cannot cover are
skipped gracefully.
"""

from __future__ import annotations

import numpy as np

from ..ephemeris.calendar import format_jd
from .inference import DecoupledInference, jd_from_timestamp


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


def create_app(ckpt_path: str, device: str = "", bank_size: int = 64,
               ephe_path: str | None = None, jpl_file: str | None = None):
    from fastapi import Body, FastAPI, Response
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse

    engine = DecoupledInference.from_checkpoint(
        ckpt_path, device=device, ephe_path=ephe_path, jpl_file=jpl_file)
    bank_jds, bank_vecs = _build_latent_bank(engine, bank_size)

    app = FastAPI(title="Decoupled Projection Engine", version="1.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                       allow_headers=["*"])

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "device": str(engine.device),
                "tension_dim": engine.cfg.sky.tension_dim,
                "bodies": list(engine.sky.body_names()),
                "latent_bank": len(bank_jds)}

    @app.get("/api/texture")
    def texture(timestamp: str, width: int = 512, height: int = 256):
        tex = engine.global_texture(timestamp, width=width, height=height)
        return Response(content=tex["bytes"], media_type="application/octet-stream",
                        headers={"X-Width": str(width), "X-Height": str(height),
                                 "X-Channels": "3", "X-JD": str(tex["jd"])})

    @app.post("/api/point")
    def point(body: dict = Body(...)) -> JSONResponse:
        try:
            r = engine.pinpoint(body["timestamp"], float(body["lat"]),
                                float(body["lon"]))
        except (KeyError, ValueError) as exc:
            return JSONResponse({"error": f"bad request: {exc}"}, status_code=400)
        return JSONResponse({
            "jd": r["jd"], "lat": r["lat"], "lon": r["lon"],
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

    return app


def serve(ckpt_path: str, host: str = "127.0.0.1", port: int = 8100,
          device: str = "", **kw) -> int:
    try:
        import uvicorn
    except ImportError:
        print('ERROR: uvicorn not installed. `pip install "kalachakra[serve]"`.')
        return 2
    app = create_app(ckpt_path, device=device, **kw)
    print(f"\nDecoupled engine API on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
    return 0
