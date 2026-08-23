"""Binary framing + FastAPI control-plane / WebSocket tests."""

import numpy as np
import pytest

from kalachakra.serving import binary


# --- binary framing (no optional deps) -----------------------------------
def test_pack_unpack_roundtrip():
    n = 20
    rng = np.random.default_rng(0)
    lat = rng.uniform(-90, 90, n).astype(np.float32)
    lng = rng.uniform(-180, 180, n).astype(np.float32)
    pot = rng.random(n).astype(np.float32)
    shear = rng.random(n).astype(np.float32)
    macro = rng.integers(0, 64, n).astype(np.uint16)
    micro = rng.integers(0, 64, n).astype(np.uint16)
    buf = binary.pack_frame(lat, lng, pot, shear, macro, micro)
    out = binary.unpack_frame(buf)
    assert np.allclose(out["lat"], lat) and np.array_equal(out["macro"], macro)
    assert "latent" not in out


def test_pack_unpack_with_latent():
    n, d = 8, 64
    latent = np.random.default_rng(1).normal(size=(n, d)).astype(np.float32)
    z = np.zeros(n, np.float32)
    m = np.zeros(n, np.uint16)
    out = binary.unpack_frame(binary.pack_frame(z, z, z, z, m, m, latent=latent))
    assert out["latent"].shape == (n, d)
    assert np.allclose(out["latent"], latent)


def test_bad_magic_rejected():
    with pytest.raises(ValueError):
        binary.unpack_frame(b"\x00" * 32)


# --- FastAPI app over a small Parquet index ------------------------------
pa = pytest.importorskip("pyarrow")
duck = pytest.importorskip("duckdb")
fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient           # noqa: E402

from kalachakra.geo import h3index                  # noqa: E402
from kalachakra.serving.app import create_app       # noqa: E402
from kalachakra.storage.parquet_store import ParquetTokenStore  # noqa: E402


def _build_index(root):
    from kalachakra.grid import geodesic
    rng = np.random.default_rng(0)
    n = 200
    frames = np.arange(n)
    jd = 2451545.0 + frames * (24 / 86400)
    grid = geodesic.fibonacci_sphere(8)
    node = rng.integers(0, 8, n)
    lat = np.rad2deg(grid.lat[node]); lng = np.rad2deg(grid.lon[node])
    macro = rng.integers(0, 64, n); micro = rng.integers(0, 64, n)
    store = ParquetTokenStore(root)
    store.write_frames({
        "jd": jd, "frame": frames.astype(np.int64), "node": node.astype(np.int32),
        "lat": lat.astype(np.float32), "lng": lng.astype(np.float32),
        "h3": h3index.cells_for_grid(lat, lng, 4),
        "macro": macro.astype(np.int16), "micro": micro.astype(np.int16),
        "leaf": (macro * 64 + micro).astype(np.int32),
        "rarity": rng.random(n).astype(np.float32),
        "potential": rng.random(n).astype(np.float32),
        "shear": rng.random(n).astype(np.float32),
    })
    return store


def test_health_and_inspect(tmp_path):
    _build_index(tmp_path / "idx")
    client = TestClient(create_app(str(tmp_path / "idx")))

    h = client.get("/health").json()
    assert h["status"] == "ok" and h["tier1"] is True

    resp = client.post("/inspect", json={
        "min_lat": -90, "min_lng": -180, "max_lat": 90, "max_lng": 180,
        "start": "2000-01-01T12:00:00Z", "end": "2000-01-01T15:00:00Z",
        "velocity": 1.0, "rarity_min": 0.0, "limit": 20,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_rows"] > 0 and body["tier"] in ("tier1", "tier2", "tier3")
    assert set(body["band_gains"]) == {"micro", "fast", "cyclic", "macro"}
    assert "latent" not in body["rows"][0]      # heavy field excluded from JSON


def test_microgrid_and_telemetry(tmp_path):
    from kalachakra.ephemeris import global_state
    if not global_state.ephemeris_available():
        pytest.skip("pyswisseph not installed")
    _build_index(tmp_path / "idx")
    client = TestClient(create_app(str(tmp_path / "idx")))

    # §4 dynamic micro-grid over a region, at the 2024 eclipse instant.
    mg = client.post("/microgrid", json={
        "min_lat": 20, "min_lng": 70, "max_lat": 30, "max_lng": 80,
        "datetime": "2024-04-08T18:17:00Z", "density": 12,
    }).json()
    assert mg["n_nodes"] == 144
    assert len(mg["potential"]) == 144 and mg["resolution_km"] > 0
    assert max(mg["potential"]) >= min(mg["potential"])

    # §5 Sidebar Inspector telemetry at a coordinate.
    tel = client.post("/telemetry", json={
        "lat": 27.0, "lng": 78.0, "datetime": "2024-04-08T18:17:00Z",
    }).json()
    assert set(tel["band_energies"]) == {"micro", "fast", "cyclic", "macro"}
    assert len(tel["entities"]) == 9              # 9 weighted bodies
    assert tel["eclipse"]["is_eclipse"] is True   # real total solar eclipse
    assert all("unit_vector" in b and "radial_distance_au" in b
               for b in tel["entities"])


def test_websocket_streams_binary(tmp_path):
    _build_index(tmp_path / "idx")
    client = TestClient(create_app(str(tmp_path / "idx")))
    with client.websocket_connect("/stream") as ws:
        ws.send_json({
            "min_lat": -90, "min_lng": -180, "max_lat": 90, "max_lng": 180,
            "start": "2000-01-01T12:00:00Z", "end": "2000-01-01T15:00:00Z",
            "limit": 30,
        })
        buf = ws.receive_bytes()
    frame = binary.unpack_frame(buf)
    assert frame["potential"].shape[0] > 0
    assert frame["macro"].shape == frame["potential"].shape
