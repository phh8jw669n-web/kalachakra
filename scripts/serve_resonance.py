#!/usr/bin/env python3
"""
Act II — Geo-Resonance Globe: inference + resonance service for a v3 checkpoint.

Standalone MVP testbed. Loads a trained VQ-bottleneck model
(models.autoencoder_v3), triangulates the geodesic mesh into a solid surface, and
serves both a static single-frame resonance and an animated time-series stream of
64-D latent field resonance against an anchor.

Endpoints:
  GET  /                     -> the self-contained WebGL globe (web/resonance.html)
  GET  /health               -> status + node/triangle counts
  GET  /api/topology         -> binary: [u32 nVerts][u32 nTris][verts f32 N*3]
                                [indices u32 M*3]   (a watertight sphere mesh)
  GET  /api/coastlines       -> binary: [u32 nSeg][verts f32 nSeg*2*3]
                                (continent outlines as GL_LINES, mesh xyz frame)
  GET  /api/mesh             -> binary Float32 [N*3] vertex unit-vectors (legacy)
  POST /api/resonance        -> single-frame {mags}{sims} for one anchor (legacy)
  POST /api/stream_resonance -> streamed animated frames over a timeline:
       body {anchor_lat, anchor_lon, start_date, end_date, step_hours}
       stream of frames, each: [u32 frameLen][u32 tsLen][ts utf8]
                               [magnitudes f32 N][cosine_similarities f32 N]

Magnitudes are ||z||_2 normalized by the start-frame scale (consistent across the
animation); similarities are cosine(z(t), z_anchor) with z_anchor **frozen at the
Start Date** — the whole timeline is compared against that one fixed moment. The
entire range is projected and encoded up front in batched forward passes
(``compute_resonance_series``), then the sync generator only serializes the cached
fields, so frames stream out with no per-frame inference stall. The heavy compute
runs in Starlette's threadpool (sync endpoint), so the event loop stays free, and
the accelerator cache is released between slabs to keep MPS steady.

Requires:  pip install "kalachakra[train,serve,transducer]"
           (torch + fastapi/uvicorn + scipy for the triangulation)

Example:
    python scripts/serve_resonance.py --checkpoint checkpoints/v3/model_step_000025.pt
    # then open http://localhost:8000
"""

import argparse
import struct
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


def _empty_cache(device):
    import torch
    if device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


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


def build_topology(grid):
    """Triangulate the point set into a watertight surface (convex hull == the
    Delaunay triangulation on a sphere). Returns (verts_f32[N*3], indices_u32[M*3]).

    The raw k-NN graph cannot form a valid manifold, so the hull is what actually
    yields a solid, non-overlapping mesh (the PRD's "blob" fix).

    Every triangle is oriented so its winding gives an **outward-facing** normal
    (``cross(b-a, c-a) . centroid > 0``). scipy's ``ConvexHull.simplices`` are not
    consistently wound, so this is what lets the client cull back faces
    (``THREE.FrontSide``) and hide the rear hemisphere cleanly."""
    try:
        from scipy.spatial import ConvexHull
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("scipy is required for the mesh triangulation "
                           '(`pip install "kalachakra[transducer]"`).') from exc
    pts = grid.xyz
    tris = np.asarray(ConvexHull(pts).simplices, dtype=np.int64)     # (M, 3)
    a, b, c = pts[tris[:, 0]], pts[tris[:, 1]], pts[tris[:, 2]]
    normal = np.cross(b - a, c - a)
    centroid = (a + b + c) / 3.0
    inward = np.einsum("ij,ij->i", normal, centroid) < 0.0          # normal points in
    swap = tris[inward, 1].copy()                                   # -> reverse winding
    tris[inward, 1] = tris[inward, 2]
    tris[inward, 2] = swap
    verts = pts.astype("<f4").ravel()                              # (N*3,)
    return verts, tris.astype(np.uint32).ravel()


def _lonlat_to_xyz(lon_deg, lat_deg, radius):
    """GeoJSON [lon, lat] (deg) -> our mesh convention xyz on a sphere of ``radius``.

    Uses exactly the projection engine's convention
    ``xyz = [cos(lat)cos(lon), cos(lat)sin(lon), sin(lat)]`` (see nearest_node and
    grid.geodesic), so coastlines register precisely against the node field.
    """
    lon = np.deg2rad(np.asarray(lon_deg, dtype=np.float64))
    lat = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    cl = np.cos(lat)
    return np.stack([cl * np.cos(lon), cl * np.sin(lon), np.sin(lat)], axis=-1) * radius


def build_coastlines(radius: float = 0.998):
    """Continent outlines as GL_LINES vertex pairs (float32, our xyz convention).

    Reads ``web/coastlines.geojson`` (Natural Earth 110m land polygons) and turns
    every polygon ring into consecutive segment endpoints so a single
    ``THREE.LineSegments`` draws crisp, high-contrast coastlines that share the
    mesh's coordinate frame. Returns an empty array if the dataset is absent.
    """
    import json

    path = Path(__file__).resolve().parents[1] / "web" / "coastlines.geojson"
    if not path.is_file():
        return np.zeros((0,), dtype="<f4")
    data = json.loads(path.read_text())

    def rings(geom):
        t = geom.get("type")
        if t == "Polygon":
            return geom["coordinates"]
        if t == "MultiPolygon":
            return [ring for poly in geom["coordinates"] for ring in poly]
        if t == "LineString":
            return [geom["coordinates"]]
        if t == "MultiLineString":
            return geom["coordinates"]
        return []

    segs = []
    for feat in data.get("features", []):
        for ring in rings(feat.get("geometry", {})):
            pts = np.asarray(ring, dtype=np.float64)
            if pts.shape[0] < 2:
                continue
            xyz = _lonlat_to_xyz(pts[:, 0], pts[:, 1], radius)     # (K, 3)
            pairs = np.empty((xyz.shape[0] - 1, 2, 3), dtype=np.float64)
            pairs[:, 0, :] = xyz[:-1]
            pairs[:, 1, :] = xyz[1:]
            segs.append(pairs.reshape(-1, 3))
    if not segs:
        return np.zeros((0,), dtype="<f4")
    return np.concatenate(segs, axis=0).astype("<f4").ravel()


def latent_at(model, grid, jd: float, window: int, device):
    """Continuous latent z (N, 64) CPU tensor at Julian Day ``jd`` (window center)."""
    import torch

    from kalachakra.ephemeris import global_state, timeline
    from kalachakra.projection import spatial

    center = int(timeline.jd_to_frame(jd))
    idx = np.arange(center - window // 2, center - window // 2 + window)
    jds = timeline.frame_to_jd(idx)
    fields = []
    for j in jds:
        g = global_state.global_state_frame(float(j))
        fields.append(spatial.project(g, float(j), grid).reshape(grid.n_nodes, -1))
    e = np.stack(fields, axis=0)[None]                     # (1, T, N, 50)
    e_t = torch.from_numpy(e.astype(np.float32)).to(device)
    with torch.no_grad():
        z_seq = model.encode(e_t)[0]                        # (T, N, 64)
    return z_seq[window // 2].float().cpu()                 # (N, 64)


def _project_fields(grid, jds):
    """Project a list of Julian Days to local fields, shape ``(len(jds), N, 50)``."""
    from kalachakra.ephemeris import global_state
    from kalachakra.projection import spatial

    out = []
    for j in jds:
        g = global_state.global_state_frame(float(j))
        out.append(spatial.project(g, float(j), grid).reshape(grid.n_nodes, -1))
    return np.stack(out, axis=0)


def auto_encode_batch(n_nodes: int) -> int:
    """Timeline steps per encoder pass that keep a full-mesh activation bounded.

    Each step is one row of a ``(Tb, 1, N, hidden)`` activation, so peak memory
    scales with ``Tb * N``; ~1.5M node-rows/pass stays well within a laptop's MPS
    budget at N=122,880 (=> ~12) while letting small test meshes batch freely.
    """
    return int(max(2, min(64, 1_500_000 // max(n_nodes, 1))))


def compute_resonance_series(model, grid, jds, node: int, device, encode_batch: int):
    """Batch pre-computation of the whole animation (no per-frame inference).

    Projects every requested timeline step once, encodes them in batched passes
    (each slab is ``(Tb, 1, N, 50)`` -> ``(Tb, N, 64)`` in a single forward), and
    reduces to per-frame magnitude/similarity fields. The anchor latent
    ``z_anchor`` and the magnitude reference scale are **frozen from the first
    frame (Start Date)** and reused for every later step, so the animation
    compares a changing field against one fixed moment in the past.

    Returns ``(mags[T, N] f32 in [0, 1.5], sims[T, N] f32 in [-1, 1],
    anchor_norm, ref_scale)``.
    """
    import torch

    n = grid.n_nodes
    t_total = len(jds)
    mags = np.empty((t_total, n), dtype="<f4")
    sims = np.empty((t_total, n), dtype="<f4")
    z_anchor = None
    ref = 1.0
    for s in range(0, t_total, encode_batch):
        e = min(s + encode_batch, t_total)
        fields = _project_fields(grid, jds[s:e])[:, None]      # (Tb, 1, N, 50)
        e_t = torch.from_numpy(fields.astype(np.float32)).to(device)
        with torch.no_grad():
            z = model.encode(e_t)[:, 0]                         # (Tb, N, 64)
        if z_anchor is None:                                    # freeze from frame 0
            z_anchor = z[0, node].clone()
            ref = float(z[0].norm(dim=1).max()) + 1e-12
        m = (z.norm(dim=2) / ref).clamp(0.0, 1.5)              # (Tb, N)
        sm = torch.nn.functional.cosine_similarity(
            z, z_anchor.view(1, 1, -1), dim=2)                  # (Tb, N)
        mags[s:e] = m.cpu().numpy().astype("<f4")
        sims[s:e] = sm.cpu().numpy().astype("<f4")
        del z, e_t
        _empty_cache(device)
    return mags, sims, float(z_anchor.norm()), ref


def nearest_node(grid, lat_deg: float, lon_deg: float) -> int:
    la, lo = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    target = np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
    return int(np.argmax(grid.xyz @ target))


def create_app(checkpoint: str, date: str = "now", window: int = 32,
               stream_window: int = 16, max_frames: int = 400,
               encode_batch: int = 0, device=None):
    """Build the FastAPI resonance app.

    ``encode_batch`` is the number of timeline steps encoded per forward pass in
    the streamed animation (0 = auto-size from the node count); ``stream_window``
    is retained for backward compatibility but the batched stream no longer uses
    a per-frame window.
    """
    import torch
    from fastapi import Body, FastAPI, Request, Response
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, StreamingResponse

    from kalachakra.ephemeris import global_state
    from kalachakra.ephemeris.calendar import format_jd, parse_datetime

    device = device or _select_device()
    if global_state.ephemeris_available():
        global_state.auto_configure()

    model, cfg, grid = load_model_and_grid(checkpoint, device)
    N = grid.n_nodes
    enc_batch = encode_batch if encode_batch and encode_batch > 0 else auto_encode_batch(N)
    print(f"Loaded v3 model on {device}: {N:,} nodes, latent={cfg.latent}, "
          f"codebook={cfg.codebook_size}, encode_batch={enc_batch}")

    print("Triangulating mesh surface (convex hull)...")
    verts, indices = build_topology(grid)
    n_tris = indices.size // 3
    topo = (struct.pack("<II", N, n_tris) + verts.tobytes() + indices.tobytes())
    print(f"  {n_tris:,} triangles.")

    coast = build_coastlines()
    n_coast_seg = coast.size // 6                         # 2 endpoints * 3 floats
    coast_bytes = struct.pack("<I", n_coast_seg) + coast.tobytes()
    print(f"  {n_coast_seg:,} coastline segments." if n_coast_seg
          else "  (no coastline dataset — graticule only)")

    print(f"Computing initial latent field for {date} (window={window})...")
    z0 = latent_at(model, grid, parse_datetime(date), window, device)
    mags0 = z0.norm(dim=1)
    lo0, hi0 = float(mags0.min()), float(mags0.max())
    mags0_norm = ((mags0 - lo0) / (hi0 - lo0 + 1e-12)).numpy().astype("<f4")
    print(f"  ready. ||z|| range [{lo0:.3f}, {hi0:.3f}].")

    app = FastAPI(title="Kalachakra Geo-Resonance Globe", version="2.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                       allow_headers=["*"], expose_headers=["*"])

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "n_nodes": N, "n_triangles": n_tris,
                "n_coastline_segments": n_coast_seg,
                "latent": cfg.latent, "codebook": cfg.codebook_size}

    @app.get("/api/topology")
    def topology() -> Response:
        return Response(content=topo, media_type="application/octet-stream",
                        headers={"X-N-Nodes": str(N), "X-N-Tris": str(n_tris)})

    @app.get("/api/coastlines")
    def coastlines() -> Response:
        return Response(content=coast_bytes, media_type="application/octet-stream",
                        headers={"X-N-Segments": str(n_coast_seg)})

    @app.get("/api/mesh")
    def mesh() -> Response:
        return Response(content=verts.tobytes(), media_type="application/octet-stream",
                        headers={"X-N-Nodes": str(N)})

    @app.post("/api/resonance")
    async def resonance(req: Request) -> Response:
        body = await req.json()
        if body.get("anchor_node_id") is not None:
            node = int(body["anchor_node_id"]) % N
        else:
            node = nearest_node(grid, float(body.get("anchor_lat", 0.0)),
                                float(body.get("anchor_lon", 0.0)))
        sims = torch.nn.functional.cosine_similarity(
            z0, z0[node].unsqueeze(0), dim=1).numpy().astype("<f4")
        payload = mags0_norm.tobytes() + sims.tobytes()
        return Response(content=payload, media_type="application/octet-stream",
                        headers={"X-Node-Id": str(node),
                                 "X-Sim-Min": f"{float(sims.min()):.6f}",
                                 "X-Sim-Max": f"{float(sims.max()):.6f}",
                                 "X-Anchor-Norm": f"{float(z0[node].norm()):.6f}",
                                 "X-Anchor-Lat": f"{float(np.rad2deg(grid.lat[node])):.4f}",
                                 "X-Anchor-Lon": f"{float(np.rad2deg(grid.lon[node])):.4f}",
                                 "X-N-Nodes": str(N)})

    def _timeline(j0: float, j1: float, step_hours: float):
        step_days = max(step_hours, 1e-6) / 24.0
        n = int(np.floor((j1 - j0) / step_days)) + 1 if j1 > j0 else 1
        if n > max_frames:                                  # keep the run bounded
            step_days = (j1 - j0) / (max_frames - 1)
            n = max_frames
        return [j0 + k * step_days for k in range(max(n, 1))]

    @app.post("/api/stream_resonance")
    def stream_resonance(body: dict = Body(...)) -> StreamingResponse:
        lat = float(body.get("anchor_lat", 0.0))
        lon = float(body.get("anchor_lon", 0.0))
        node = (int(body["anchor_node_id"]) % N if body.get("anchor_node_id") is not None
                else nearest_node(grid, lat, lon))
        j0 = parse_datetime(body.get("start_date", date))
        j1 = parse_datetime(body.get("end_date", body.get("start_date", date)))
        step_hours = float(body.get("step_hours", 24.0))
        jds = _timeline(j0, j1, step_hours)
        anchor_date = format_jd(float(jds[0]))

        # Batch pre-computation: project + encode the entire range up front, with
        # z_anchor frozen from the Start Date. The heavy PyTorch work happens here
        # (sync endpoint -> Starlette threadpool, so the event loop is free); the
        # generator below only serializes the cached fields, so frames then stream
        # out as fast as the socket drains.
        mags, sims, anchor_norm, _ref = compute_resonance_series(
            model, grid, jds, node, device, enc_batch)

        def frames():
            for k in range(len(jds)):
                ts = format_jd(float(jds[k])).encode("utf-8")
                body_bytes = (struct.pack("<I", len(ts)) + ts
                              + mags[k].tobytes() + sims[k].tobytes())
                yield struct.pack("<I", len(body_bytes)) + body_bytes

        return StreamingResponse(frames(), media_type="application/octet-stream",
                                 headers={"X-N-Nodes": str(N),
                                          "X-N-Frames": str(len(jds)),
                                          "X-Node-Id": str(node),
                                          "X-Anchor-Date": anchor_date,
                                          "X-Anchor-Norm": f"{anchor_norm:.6f}"})

    @app.get("/")
    def index():
        if _WEB.is_file():
            return FileResponse(str(_WEB))
        return Response("web/resonance.html not found", status_code=404)

    app.state.n_nodes = N
    app.state.n_triangles = n_tris
    app.state.n_coastline_segments = n_coast_seg
    return app


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="checkpoints/v3/model_step_000025.pt")
    p.add_argument("--date", default="now", help="initial frame (UTC ISO or 'now')")
    p.add_argument("--window", type=int, default=32, help="encode window for the static frame")
    p.add_argument("--stream-window", type=int, default=16,
                   help="legacy; unused by the batched stream")
    p.add_argument("--encode-batch", type=int, default=0,
                   help="timeline steps per encoder forward pass (0 = auto from node count)")
    p.add_argument("--max-frames", type=int, default=400, help="cap on timeline frames")
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
    app = create_app(args.checkpoint, date=args.date, window=args.window,
                     stream_window=args.stream_window, max_frames=args.max_frames,
                     encode_batch=args.encode_batch)
    print(f"\nGeo-Resonance Globe on http://{args.host}:{args.port}  (open it in a browser)")
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
