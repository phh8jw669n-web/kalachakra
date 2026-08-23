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


def _build_index_with_latents(root, n=60):
    """Index whose latents form two well-separated blobs, so clustering finds them."""
    from kalachakra.grid import geodesic
    rng = np.random.default_rng(3)
    frames = np.arange(n)
    jd = 2451545.0 + frames * (24 / 86400)
    grid = geodesic.fibonacci_sphere(8)
    node = rng.integers(0, 8, n)
    lat = np.rad2deg(grid.lat[node]); lng = np.rad2deg(grid.lon[node])
    macro = rng.integers(0, 64, n); micro = rng.integers(0, 64, n)
    half = n // 2
    centers = np.zeros((n, 64), np.float32)
    centers[:half] = 10.0                          # blob A
    centers[half:] = -10.0                         # blob B
    latent = (centers + rng.normal(scale=0.1, size=(n, 64))).astype(np.float32)
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
        "latent": latent,
    })
    return store


def test_inspect_clusters_latents(tmp_path):
    _build_index_with_latents(tmp_path / "idx")
    client = TestClient(create_app(str(tmp_path / "idx")))
    body = client.post("/inspect", json={
        "min_lat": -90, "min_lng": -180, "max_lat": 90, "max_lng": 180,
        "start": "2000-01-01T12:00:00Z", "end": "2000-01-01T13:00:00Z",
        "limit": 100, "cluster_min_size": 3,
    }).json()
    assert body["cluster_method"] in ("hdbscan", "fallback")
    assert body["n_clusters"] >= 1                       # two blobs are separable
    assert all("cluster_id" in r for r in body["rows"])
    assert body["global_latent"] is not None and len(body["global_latent"]) == 64
    assert "latent" not in body["rows"][0]               # heavy field still excluded


def test_news_cards(tmp_path):
    from kalachakra.ephemeris import global_state
    if not global_state.ephemeris_available():
        pytest.skip("pyswisseph not installed")
    _build_index(tmp_path / "idx")
    client = TestClient(create_app(str(tmp_path / "idx")))
    news = client.post("/news", json={
        "min_lat": -90, "min_lng": -180, "max_lat": 90, "max_lng": 180,
        "start": "2000-01-01T12:00:00Z", "end": "2000-01-01T15:00:00Z",
        "rarity_min": 0.0, "top": 5,
    }).json()
    assert news["n"] >= 1 and len(news["cards"]) == news["n"]
    card = news["cards"][0]
    assert set(("jd", "lat", "lng", "macro", "micro", "rarity",
                "rarity_percentile", "applying", "bodies")) <= set(card)
    assert len(card["bodies"]) == 9                      # 9 weighted bodies
    assert all("unit_vector" in b for b in card["bodies"])
    # cards are ordered rarest-first
    rar = [c["rarity"] for c in news["cards"]]
    assert rar == sorted(rar, reverse=True)


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


def test_cors_enabled(tmp_path):
    _build_index(tmp_path / "idx")
    client = TestClient(create_app(str(tmp_path / "idx")))
    r = client.get("/health", headers={"Origin": "http://example.com"})
    assert r.headers.get("access-control-allow-origin") == "*"


def test_web_ui_mounts_and_serves_visuals(tmp_path):
    from pathlib import Path

    from kalachakra.serving.webui import default_web_dir, mount_web_ui
    web = default_web_dir()
    if not web.is_dir():                       # pragma: no cover
        pytest.skip("repo web/ dir not present")
    _build_index(tmp_path / "idx")
    app = create_app(str(tmp_path / "idx"))
    assert mount_web_ui(app, web) == web
    client = TestClient(app)
    # Both visuals are served same-origin, so the browser needs no CORS at all.
    assert client.get("/ui/radar.html").status_code == 200
    assert client.get("/ui/index.html").status_code == 200
    assert Path(web, "radar.html").exists()


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
