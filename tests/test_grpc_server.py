"""gRPC CosmicWeather service — real in-process round trip (blueprint §7.2).

Skips cleanly when grpcio / index deps are absent; when present it stands up a
real server on an ephemeral port and exercises Health / Inspect / Telemetry over
the wire, proving the typed gRPC surface mirrors the REST contract.
"""

import numpy as np
import pytest

pytest.importorskip("grpc")
pytest.importorskip("pyarrow")
pytest.importorskip("duckdb")

from kalachakra.geo import h3index                              # noqa: E402
from kalachakra.serving.grpc_server import grpc_available, serve  # noqa: E402
from kalachakra.storage.parquet_store import ParquetTokenStore   # noqa: E402


def _build_index(root, n=120):
    from kalachakra.grid import geodesic
    rng = np.random.default_rng(0)
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


@pytest.fixture()
def channel_and_stubs(tmp_path):
    import grpc
    from kalachakra.serving.grpc_gen import kalachakra_pb2 as pb
    from kalachakra.serving.grpc_gen import kalachakra_pb2_grpc as pbg

    _build_index(tmp_path / "idx")
    server = serve(str(tmp_path / "idx"), host="127.0.0.1", port=0)
    port = server._kalachakra_port
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    stub = pbg.CosmicWeatherStub(channel)
    try:
        yield stub, pb
    finally:
        channel.close()
        server.stop(grace=None)


def test_grpc_available():
    assert grpc_available() is True


def test_health(channel_and_stubs):
    stub, pb = channel_and_stubs
    reply = stub.Health(pb.HealthRequest())
    assert reply.status == "ok" and reply.tier1 is True


def test_inspect_roundtrip(channel_and_stubs):
    stub, pb = channel_and_stubs
    reply = stub.Inspect(pb.InspectRequest(
        min_lat=-90, min_lng=-180, max_lat=90, max_lng=180,
        start="2000-01-01T12:00:00Z", end="2000-01-01T15:00:00Z",
        velocity=1.0, rarity_min=0.0, limit=20))
    assert reply.n_rows > 0 and reply.tier in ("tier1", "tier2", "tier3")
    assert set(reply.band_gains) == {"micro", "fast", "cyclic", "macro"}
    rar = [r.rarity for r in reply.rows]
    assert rar == sorted(rar, reverse=True)             # ORDER BY rarity DESC


def test_telemetry_roundtrip(channel_and_stubs):
    from kalachakra.ephemeris import global_state
    if not global_state.ephemeris_available():
        pytest.skip("pyswisseph not installed")
    stub, pb = channel_and_stubs
    reply = stub.Telemetry(pb.TelemetryRequest(
        lat=27.0, lng=78.0, datetime="2024-04-08T18:17:00Z"))
    assert set(reply.band_energies) == {"micro", "fast", "cyclic", "macro"}
    assert len(reply.entities) == 9                     # 9 weighted bodies
    assert reply.is_eclipse is True                     # real total solar eclipse
    assert all(len(b.unit_vector) == 3 for b in reply.entities)
