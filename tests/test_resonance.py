"""Act II Geo-Resonance backend: checkpoint load -> inference -> endpoints.

Builds a tiny v3 checkpoint (in the exact format scripts/train_v3.py saves), then
exercises serve_resonance.create_app over it: /health, binary /api/mesh, binary
/api/resonance with its stats headers, the triangulated /api/topology surface, and
the animated /api/stream_resonance temporal stream.
"""

import importlib.util
import struct
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient                             # noqa: E402

from kalachakra.ephemeris import global_state                        # noqa: E402
from kalachakra.grid import geodesic                                 # noqa: E402
from kalachakra.models.autoencoder_v3 import (                       # noqa: E402
    VQAutoencoderV3, VQAutoencoderV3Config, build_knn,
)


def _load_serve():
    p = Path(__file__).resolve().parents[1] / "scripts" / "serve_resonance.py"
    spec = importlib.util.spec_from_file_location("serve_resonance", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tiny_checkpoint(path, n=400):
    grid = geodesic.fibonacci_sphere(n)
    nb = build_knn(grid.xyz, 7)
    cfg = VQAutoencoderV3Config(n_nodes=n, in_features=50, hidden=16, latent=64,
                                fourier_modes=6, knn=7, n_blocks=2, codebook_size=4096,
                                node_chunk=128, vq_chunk=4096)
    model = VQAutoencoderV3(cfg, nb)
    torch.save({
        "format": "kalachakra-vqmodel-v3",
        "projection_version": 2,
        "config": asdict(cfg),
        "neighbors": np.asarray(nb, dtype=np.int64),
        "grid_xyz": np.asarray(grid.xyz, dtype=np.float64),
        "state_dict": model.state_dict(),
        "step": 25,
    }, path)
    return n


def test_resonance_backend_end_to_end(tmp_path):
    if not global_state.ephemeris_available():
        pytest.skip("pyswisseph not installed")
    serve = _load_serve()
    ckpt = tmp_path / "model_step_000025.pt"
    n = _tiny_checkpoint(ckpt)

    app = serve.create_app(str(ckpt), date="2024-04-08T18:17:00Z", window=8,
                           device=torch.device("cpu"))
    client = TestClient(app)

    h = client.get("/health").json()
    assert h["status"] == "ok" and h["n_nodes"] == n and h["latent"] == 64

    # /api/mesh: N*3 float32 unit vectors.
    mesh = client.get("/api/mesh")
    assert mesh.status_code == 200
    verts = np.frombuffer(mesh.content, dtype="<f4")
    assert verts.size == n * 3
    assert int(mesh.headers["X-N-Nodes"]) == n
    # unit vectors
    v = verts.reshape(n, 3)
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-4)

    # /api/resonance: N magnitudes + N similarities, plus stats headers.
    r = client.post("/api/resonance", json={"anchor_lat": 26.9, "anchor_lon": 75.7})
    assert r.status_code == 200
    arr = np.frombuffer(r.content, dtype="<f4")
    assert arr.size == 2 * n
    mags, sims = arr[:n], arr[n:]
    assert mags.min() >= 0.0 and mags.max() <= 1.0 + 1e-5        # normalized [0,1]
    assert sims.min() >= -1.0 - 1e-4 and sims.max() <= 1.0 + 1e-4
    node = int(r.headers["X-Node-Id"])
    assert 0 <= node < n
    assert abs(sims[node] - 1.0) < 1e-4                          # anchor ~ perfectly self-similar
    assert -1.0 <= float(r.headers["X-Sim-Min"]) <= 1.0


def test_resonance_node_id_and_nearest(tmp_path):
    serve = _load_serve()
    if not global_state.ephemeris_available():
        pytest.skip("pyswisseph not installed")
    ckpt = tmp_path / "m.pt"
    _tiny_checkpoint(ckpt, n=300)
    app = serve.create_app(str(ckpt), date="2024-01-01T00:00:00Z", window=6,
                           device=torch.device("cpu"))
    client = TestClient(app)
    # explicit node id round-trips
    r = client.post("/api/resonance", json={"anchor_node_id": 42})
    assert int(r.headers["X-Node-Id"]) == 42
    # nearest-node lookup is deterministic
    a = client.post("/api/resonance", json={"anchor_lat": 0.0, "anchor_lon": 0.0})
    b = client.post("/api/resonance", json={"anchor_lat": 0.0, "anchor_lon": 0.0})
    assert a.headers["X-Node-Id"] == b.headers["X-Node-Id"]


def test_topology_is_watertight_hull(tmp_path):
    if not global_state.ephemeris_available():
        pytest.skip("pyswisseph not installed")
    pytest.importorskip("scipy")
    serve = _load_serve()
    ckpt = tmp_path / "model_step_000025.pt"
    n = _tiny_checkpoint(ckpt)
    app = serve.create_app(str(ckpt), date="2024-04-08T18:17:00Z", window=8,
                           device=torch.device("cpu"))
    client = TestClient(app)

    h = client.get("/health").json()
    assert h["n_triangles"] > 0

    res = client.get("/api/topology")
    assert res.status_code == 200
    N, n_tris = struct.unpack("<II", res.content[:8])
    assert N == n and n_tris == int(res.headers["X-N-Tris"])
    # exact binary layout: [u32 N][u32 nTris][verts f32 N*3][indices u32 M*3]
    assert len(res.content) == 8 + N * 12 + n_tris * 12

    verts = np.frombuffer(res.content, dtype="<f4", count=N * 3, offset=8).reshape(N, 3)
    assert np.allclose(np.linalg.norm(verts, axis=1), 1.0, atol=1e-4)   # unit sphere
    indices = np.frombuffer(res.content, dtype="<u4", offset=8 + N * 12).reshape(n_tris, 3)
    assert indices.min() >= 0 and indices.max() < N                     # valid refs
    # a closed hull of a sphere: Euler characteristic V - E + F = 2, and every
    # edge is shared by exactly two triangles => E = 3F/2, so F = 2N - 4.
    assert n_tris == 2 * N - 4
    # every triangle references three distinct vertices
    assert (indices[:, 0] != indices[:, 1]).all()
    assert (indices[:, 1] != indices[:, 2]).all()
    assert (indices[:, 0] != indices[:, 2]).all()

    # every triangle is wound outward (cross(b-a, c-a) . centroid > 0), so the
    # client can cull back faces (THREE.FrontSide) and hide the rear hemisphere.
    a = verts[indices[:, 0]]
    b = verts[indices[:, 1]]
    c = verts[indices[:, 2]]
    normal = np.cross(b - a, c - a)
    centroid = (a + b + c) / 3.0
    assert (np.einsum("ij,ij->i", normal, centroid) > 0.0).all()


def test_coastlines_overlay(tmp_path):
    if not global_state.ephemeris_available():
        pytest.skip("pyswisseph not installed")
    serve = _load_serve()
    ckpt = tmp_path / "model_step_000025.pt"
    _tiny_checkpoint(ckpt)
    app = serve.create_app(str(ckpt), date="2024-04-08T18:17:00Z", window=8,
                           device=torch.device("cpu"))
    client = TestClient(app)

    h = client.get("/health").json()
    n_seg = h["n_coastline_segments"]
    assert n_seg > 100                       # a real world outline, not a stub

    res = client.get("/api/coastlines")
    assert res.status_code == 200
    (hdr_seg,) = struct.unpack("<I", res.content[:4])
    assert hdr_seg == n_seg == int(res.headers["X-N-Segments"])
    # exact layout: [u32 nSeg][verts f32 nSeg*2*3]
    assert len(res.content) == 4 + n_seg * 6 * 4

    verts = np.frombuffer(res.content, dtype="<f4", offset=4).reshape(n_seg * 2, 3)
    # coastlines sit on the reference shell just above the Earth sphere
    assert np.allclose(np.linalg.norm(verts, axis=1), 0.998, atol=1e-3)
    # and they span the whole globe (both hemispheres in every axis)
    assert verts[:, 2].min() < -0.5 and verts[:, 2].max() > 0.5

    # convention check: the builder matches the mesh's lon/lat->xyz exactly
    xyz = serve._lonlat_to_xyz(0.0, 0.0, 1.0)          # Greenwich / equator
    assert np.allclose(xyz, [1.0, 0.0, 0.0], atol=1e-9)
    xyz = serve._lonlat_to_xyz(90.0, 0.0, 1.0)         # 90E on the equator -> +y
    assert np.allclose(xyz, [0.0, 1.0, 0.0], atol=1e-9)


def test_stream_resonance_frames(tmp_path):
    if not global_state.ephemeris_available():
        pytest.skip("pyswisseph not installed")
    serve = _load_serve()
    ckpt = tmp_path / "model_step_000025.pt"
    n = _tiny_checkpoint(ckpt)
    app = serve.create_app(str(ckpt), date="2024-01-01T00:00:00Z", window=8,
                           stream_window=6, device=torch.device("cpu"))
    client = TestClient(app)

    r = client.post("/api/stream_resonance", json={
        "anchor_lat": 26.9, "anchor_lon": 75.7,
        "start_date": "2024-01-01T00:00:00Z", "end_date": "2024-01-04T00:00:00Z",
        "step_hours": 24.0})
    assert r.status_code == 200
    declared = int(r.headers["X-N-Frames"])
    node = int(r.headers["X-Node-Id"])
    assert 0 <= node < n
    assert declared == 4                       # [0,1,2,3] days inclusive

    # parse the frame stream exactly as the browser does
    data = r.content
    off, frames = 0, []
    while off + 4 <= len(data):
        (flen,) = struct.unpack_from("<I", data, off)
        off += 4
        fb = data[off:off + flen]
        off += flen
        (ts_len,) = struct.unpack_from("<I", fb, 0)
        ts = fb[4:4 + ts_len].decode("utf-8")
        base = 4 + ts_len
        mags = np.frombuffer(fb, dtype="<f4", count=n, offset=base)
        sims = np.frombuffer(fb, dtype="<f4", count=n, offset=base + n * 4)
        assert flen == 4 + ts_len + 8 * n
        assert len(ts) > 0
        assert mags.min() >= 0.0 and mags.max() <= 1.5 + 1e-5
        assert sims.min() >= -1.0 - 1e-4 and sims.max() <= 1.0 + 1e-4
        frames.append((ts, mags, sims))

    assert off == len(data)                    # no trailing bytes
    assert len(frames) == declared
    # the anchor is self-similar in the first frame (z_anchor is taken from it)
    assert abs(frames[0][2][node] - 1.0) < 1e-4
    # frames advance in time (timestamps differ across the run)
    assert frames[0][0] != frames[-1][0]


def test_stream_anchor_is_frozen_to_start(tmp_path):
    """Fix #1: z_anchor is frozen at the Start Date, not re-derived per frame.

    If the anchor were re-derived each step, the anchor node's similarity would be
    1.0 in *every* frame. With a frozen anchor it is 1.0 only at frame 0 and drifts
    as the field evolves against that fixed past moment.
    """
    if not global_state.ephemeris_available():
        pytest.skip("pyswisseph not installed")
    serve = _load_serve()
    ckpt = tmp_path / "model_step_000025.pt"
    n = _tiny_checkpoint(ckpt)
    app = serve.create_app(str(ckpt), date="2024-01-01T00:00:00Z", window=8,
                           device=torch.device("cpu"))
    client = TestClient(app)

    r = client.post("/api/stream_resonance", json={
        "anchor_lat": 26.9, "anchor_lon": 75.7,
        "start_date": "2024-01-01T00:00:00Z", "end_date": "2024-12-01T00:00:00Z",
        "step_hours": 730.0})                   # ~monthly over a year: field moves a lot
    assert r.status_code == 200
    assert r.headers["X-Anchor-Date"].startswith("2024-01-01")
    node = int(r.headers["X-Node-Id"])

    data, off, anchor_sims = r.content, 0, []
    while off + 4 <= len(data):
        (flen,) = struct.unpack_from("<I", data, off)
        off += 4
        fb = data[off:off + flen]
        off += flen
        (ts_len,) = struct.unpack_from("<I", fb, 0)
        sims = np.frombuffer(fb, dtype="<f4", count=n, offset=4 + ts_len + n * 4)
        anchor_sims.append(float(sims[node]))

    assert len(anchor_sims) >= 3
    assert abs(anchor_sims[0] - 1.0) < 1e-4              # frozen anchor == frame 0
    # a re-derived anchor would keep this at 1.0 forever; a frozen one drifts.
    assert min(anchor_sims[1:]) < 0.9
