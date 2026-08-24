#!/usr/bin/env python3
"""
Act II — Geo-Resonance Globe: inference + resonance service for a v3 checkpoint.

Standalone MVP testbed. Loads a trained VQ-bottleneck model
(models.autoencoder_v3), runs one forward pass to produce the continuous 64-D
latent z for all mesh nodes at a chosen time, and serves:

  GET  /                -> the self-contained WebGL globe (web/resonance.html)
  GET  /health          -> status + node count + timestamp
  GET  /api/mesh        -> binary Float32 [N*3] vertex unit-vectors (X-N-Nodes)
  POST /api/resonance   -> {anchor_lat, anchor_lon} or {anchor_node_id}
                           binary Float32 [N magnitudes][N similarities];
                           stats in X-Node-Id / X-Sim-Min / X-Sim-Max /
                           X-Anchor-Norm / X-Anchor-Lat / X-Anchor-Lon headers.

Magnitudes are ||z||_2 min-max normalized to [0,1]; similarities are cosine(z,
z_anchor) in [-1,1]. z is computed once at startup and cached, so changing the
anchor is instant.

Requires:  pip install "kalachakra[train,serve]"   (torch + fastapi + uvicorn)

Example:
    python scripts/serve_resonance.py --checkpoint checkpoints/v3/model_step_000025.pt
    # then open http://localhost:8000
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

_WEB = Path(__file__).resolve().parents[1] / "web" / "resonance.html"


def _select_device():
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model_and_grid(checkpoint: str, device):
    """Load a v3 checkpoint and rebuild the model + its geodesic grid."""
    import torch

    from kalachakra.grid.geodesic import Grid
    from kalachakra.models.autoencoder_v3 import VQAutoencoderV3, VQAutoencoderV3Config

    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if ck.get("format") != "kalachakra-vqmodel-v3":
        raise ValueError(f"not a v3 checkpoint (format={ck.get('format')!r}): {checkpoint}")
    cfg = VQAutoencoderV3Config(**ck["config"])
    neighbors = np.asarray(ck["neighbors"], dtype=np.int64)
    model = VQAutoencoderV3(cfg, neighbors)
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    grid_xyz = np.asarray(ck["grid_xyz"], dtype=np.float64)
    lat = np.arcsin(np.clip(grid_xyz[:, 2], -1.0, 1.0))
    lon = np.arctan2(grid_xyz[:, 1], grid_xyz[:, 0])
    grid = Grid(xyz=grid_xyz, lat=lat, lon=lon)
    return model, cfg, grid


def compute_latents(model, cfg, grid, date: str, window: int, device):
    """Project a real time-window onto the mesh, encode, return z (N, 64) on CPU.

    Uses the continuous pre-quantization latent (``encode``) at the window's
    center frame, which is what the resonance visualization inspects.
    """
    import torch

    from kalachakra.ephemeris import global_state, timeline
    from kalachakra.ephemeris.calendar import format_jd, parse_datetime
    from kalachakra.projection import spatial

    if not global_state.ephemeris_available():
        raise RuntimeError("pyswisseph is required to compute the input field.")
    global_state.auto_configure()

    jd0 = parse_datetime(date)
    center = int(timeline.jd_to_frame(jd0))
    start = center - window // 2
    idx = np.arange(start, start + window)
    jds = timeline.frame_to_jd(idx)

    fields = []
    for j in jds:
        g = global_state.global_state_frame(float(j))
        f = spatial.project(g, float(j), grid)            # (N, B, 5)
        fields.append(f.reshape(grid.n_nodes, -1))        # (N, 50)
    e = np.stack(fields, axis=0)[None]                    # (1, T, N, 50)
    e_t = torch.from_numpy(e.astype(np.float32)).to(device)

    with torch.no_grad():
        z_seq = model.encode(e_t)[0]                      # (T, N, 64)
    z = z_seq[window // 2].float().cpu()                  # (N, 64) center frame
    return z, format_jd(jd0)


def nearest_node(grid, lat_deg: float, lon_deg: float) -> int:
    """Index of the mesh node closest to a geographic coordinate."""
    la, lo = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    target = np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
    return int(np.argmax(grid.xyz @ target))


def create_app(checkpoint: str, date: str = "now", window: int = 32, device=None):
    """Build the FastAPI resonance app (z computed once, cached)."""
    import torch
    from fastapi import FastAPI, Request, Response
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse

    device = device or _select_device()
    model, cfg, grid = load_model_and_grid(checkpoint, device)
    print(f"Loaded v3 model on {device}: {grid.n_nodes:,} nodes, "
          f"latent={cfg.latent}, codebook={cfg.codebook_size}")
    print(f"Computing latent field for {date} (window={window})...")
    z, stamp = compute_latents(model, cfg, grid, date, window, device)
    mags = z.norm(dim=1)                                   # (N,)
    lo, hi = float(mags.min()), float(mags.max())
    mags_norm = ((mags - lo) / (hi - lo + 1e-12)).numpy().astype("<f4")
    verts = grid.xyz.astype("<f4").ravel()                # (N*3,)
    print(f"Latent field ready ({stamp}). ||z|| range [{lo:.3f}, {hi:.3f}].")

    app = FastAPI(title="Kalachakra Geo-Resonance Globe", version="1.0")
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
        allow_headers=["*"], expose_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "n_nodes": grid.n_nodes, "timestamp": stamp,
                "latent": cfg.latent, "codebook": cfg.codebook_size}

    @app.get("/api/mesh")
    def mesh() -> Response:
        return Response(content=verts.tobytes(), media_type="application/octet-stream",
                        headers={"X-N-Nodes": str(grid.n_nodes)})

    @app.post("/api/resonance")
    async def resonance(req: Request) -> Response:
        body = await req.json()
        if body.get("anchor_node_id") is not None:
            node = int(body["anchor_node_id"]) % grid.n_nodes
        else:
            node = nearest_node(grid, float(body.get("anchor_lat", 0.0)),
                                float(body.get("anchor_lon", 0.0)))
        z_anchor = z[node]
        sims = torch.nn.functional.cosine_similarity(
            z, z_anchor.unsqueeze(0), dim=1).numpy().astype("<f4")
        payload = mags_norm.tobytes() + sims.tobytes()
        headers = {
            "X-Node-Id": str(node),
            "X-Sim-Min": f"{float(sims.min()):.6f}",
            "X-Sim-Max": f"{float(sims.max()):.6f}",
            "X-Anchor-Norm": f"{float(z_anchor.norm()):.6f}",
            "X-Anchor-Lat": f"{float(np.rad2deg(grid.lat[node])):.4f}",
            "X-Anchor-Lon": f"{float(np.rad2deg(grid.lon[node])):.4f}",
            "X-N-Nodes": str(grid.n_nodes),
        }
        return Response(content=payload, media_type="application/octet-stream",
                        headers=headers)

    @app.get("/")
    def index():
        if _WEB.is_file():
            return FileResponse(str(_WEB))
        return Response("web/resonance.html not found", status_code=404)

    app.state.n_nodes = grid.n_nodes
    return app


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="checkpoints/v3/model_step_000025.pt")
    p.add_argument("--date", default="now", help="time frame to inspect (UTC ISO or 'now')")
    p.add_argument("--window", type=int, default=32, help="temporal window frames for encode")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not Path(args.checkpoint).exists():
        print(f"ERROR: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 2
    try:
        import uvicorn
    except Exception:
        print('ERROR: uvicorn not installed. `pip install "kalachakra[serve]"`.',
              file=sys.stderr)
        return 2
    app = create_app(args.checkpoint, date=args.date, window=args.window)
    print(f"\nGeo-Resonance Globe on http://{args.host}:{args.port}  (open it in a browser)")
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
