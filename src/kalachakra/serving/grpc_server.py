"""
gRPC control plane (blueprint §7.2) — the typed counterpart to the REST app.

Same surfaces (Health / Inspect / Telemetry) over the same DuckDB engine and
ephemeris/weather core, expressed as a strongly typed gRPC contract
(``proto/kalachakra.proto``). Kept behind a factory so importing this module never
forces ``grpcio`` (an optional ``[grpc]`` extra); the REST/WebSocket app remains
the default serving path.

    from kalachakra.serving.grpc_server import serve
    server = serve("data/index", port=50051); server.wait_for_termination()
"""

from __future__ import annotations

from concurrent import futures

import numpy as np

from ..analysis import radar, weather
from ..ephemeris import bodies, global_state
from ..ephemeris.calendar import format_jd, parse_datetime
from ..projection import spatial
from ..projection.microgrid import bbox_microgrid
from ..storage.duckdb_engine import DuckDBEngine, ViewportQuery
from ..storage.parquet_store import ParquetTokenStore
from .app import _cluster_rows


def grpc_available() -> bool:
    try:  # pragma: no cover - trivial import probe
        import grpc  # noqa: F401

        from .grpc_gen import kalachakra_pb2, kalachakra_pb2_grpc  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def _require():
    if not grpc_available():
        raise RuntimeError(
            'grpc is required. `pip install "kalachakra[grpc]"` '
            "(grpcio + grpcio-tools)."
        )


def make_servicer(store_root: str):
    """Build a CosmicWeather servicer bound to a Parquet token store."""
    _require()
    from .grpc_gen import kalachakra_pb2 as pb
    from .grpc_gen import kalachakra_pb2_grpc as pbg

    store = ParquetTokenStore(store_root)
    engine = DuckDBEngine(store)
    if global_state.ephemeris_available():
        global_state.auto_configure()

    class CosmicWeatherServicer(pbg.CosmicWeatherServicer):
        def Health(self, request, context):  # noqa: N802 (gRPC naming)
            return pb.HealthReply(
                status="ok",
                tier1=store.has_tier("tier1"),
                tier2=store.has_tier("tier2"),
                tier3=store.has_tier("tier3"),
            )

        def Inspect(self, request, context):  # noqa: N802
            j0, j1 = parse_datetime(request.start), parse_datetime(request.end)
            q = ViewportQuery(
                request.min_lat, request.min_lng, request.max_lat, request.max_lng,
                j0, j1, velocity=request.velocity or 1.0,
                rarity_min=request.rarity_min, limit=request.limit or 1000)
            rows = engine.query(q)
            span_years = max((q.end_jd - q.start_jd) / 365.25, 1e-6)
            latents = [r["latent"] for r in rows if r.get("latent") is not None]
            global_latent = (np.mean(np.asarray(latents, dtype=float), axis=0).tolist()
                             if latents else [])
            n_clusters, method = _cluster_rows(rows, latents, request.cluster_min_size)
            gains = radar.band_gains(radar.temporal_stride(
                int((q.end_jd - q.start_jd) * 86400 / 24)))
            reply = pb.InspectReply(
                tier=engine._tier_for(q), span_years=span_years,
                significance_percentile=radar.significance_percentile(span_years),
                band_gains=gains, n_rows=len(rows), global_latent=global_latent,
                n_clusters=n_clusters or 0, cluster_method=method or "")
            for r in rows:
                reply.rows.append(pb.Row(
                    jd=float(r["jd"]), lat=float(r["lat"]), lng=float(r["lng"]),
                    potential=float(r.get("potential", 0.0)),
                    shear=float(r.get("shear", 0.0)),
                    macro=int(r.get("macro", 0)), micro=int(r.get("micro", 0)),
                    rarity=float(r.get("rarity", 0.0)),
                    cluster_id=int(r.get("cluster_id", -1))))
            return reply

        def Telemetry(self, request, context):  # noqa: N802
            if not global_state.ephemeris_available():
                context.set_details("pyswisseph not installed")
                from grpc import StatusCode
                context.set_code(StatusCode.FAILED_PRECONDITION)
                return pb.TelemetryReply()
            jd = parse_datetime(request.datetime)
            g = global_state.global_state_frame(jd)
            lons = np.arctan2(g[:, 1], g[:, 0])
            af = weather.aspect_field(lons)
            sig = weather.frame_signature(jd)
            one = bbox_microgrid(request.lat - 0.01, request.lng - 0.01,
                                 request.lat + 0.01, request.lng + 0.01, 2)
            field = spatial.project(g, jd, one)
            local = float(weather.local_intensity(field, af["activation"]).mean())
            reply = pb.TelemetryReply(
                timestamp=format_jd(jd), julian_day=jd,
                lat=request.lat, lng=request.lng,
                band_energies=radar.band_energies(af["activation"], diurnal=local),
                resonance=sig.resonance, tension=sig.tension,
                geometric_potential_r=sig.potential,
                is_eclipse=bool(sig.eclipse["is_eclipse"]))
            for i, name in enumerate(bodies.NAMES):
                if weather.BODY_WEIGHTS[i] == 0:
                    continue
                reply.entities.append(pb.Body(
                    name=name,
                    unit_vector=g[i, :3].astype(float).tolist(),
                    radial_distance_au=float(g[i, 5]),
                    angular_velocity_deg_per_day=float(np.rad2deg(g[i, 3])),
                    retrograde=bool(g[i, 3] < 0)))
            return reply

    return CosmicWeatherServicer(), pb, pbg


def serve(store_root: str, host: str = "127.0.0.1", port: int = 50051,
          max_workers: int = 8):
    """Start a gRPC server over a Parquet token store; returns the running server."""
    import grpc

    servicer, _pb, pbg = make_servicer(store_root)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    pbg.add_CosmicWeatherServicer_to_server(servicer, server)
    bound = server.add_insecure_port(f"{host}:{port}")
    server.start()
    server._kalachakra_port = bound  # actual port (useful when port=0)
    return server
