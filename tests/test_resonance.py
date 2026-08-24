"""Act II Geo-Resonance backend: checkpoint load -> inference -> endpoints.

Builds a tiny v3 checkpoint (in the exact format scripts/train_v3.py saves), then
exercises serve_resonance.create_app over it: /health, binary /api/mesh, and
binary /api/resonance with its stats headers.
"""

import importlib.util
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
    n = _tiny_checkpoint(ckpt, n=300)
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
