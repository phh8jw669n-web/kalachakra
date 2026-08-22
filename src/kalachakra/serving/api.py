"""
Broadcast API and output schema (blueprint §7.2).

Defines the JSON payload contract and, when FastAPI is installed, a REST app that
exposes point queries and the full heatmap. The schema is expressed as plain
dataclasses so the contract is importable without any web framework; the gRPC
counterpart (not vendored here) mirrors the same fields.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .broadcast import BroadcastEngine, LocalReading


@dataclass
class PotentialResponse:
    """Output schema for a single ``/potential`` query (blueprint §7.2)."""

    latitude: float
    longitude: float
    frame: int
    potential_index: float
    shear_velocity: float
    cluster_id: int

    @classmethod
    def from_reading(cls, reading: LocalReading, frame: int) -> "PotentialResponse":
        return cls(
            latitude=reading.latitude_deg,
            longitude=reading.longitude_deg,
            frame=frame,
            potential_index=reading.potential_index,
            shear_velocity=reading.shear_velocity,
            cluster_id=reading.cluster_id,
        )

    def to_dict(self) -> dict:
        return asdict(self)


def create_app(engine: BroadcastEngine, frame: int = 0):
    """Build a FastAPI app over a prepared :class:`BroadcastEngine`.

    Requires ``fastapi``. Kept behind a function so importing this module never
    forces a web dependency.
    """
    try:
        from fastapi import FastAPI
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("fastapi is required for the REST broadcast app") from exc

    app = FastAPI(title="Kalachakra Cosmic Weather Broadcast", version="0.1.0")

    @app.get("/potential")
    def potential(lat: float, lon: float) -> dict:
        reading = engine.query(lat, lon)
        return PotentialResponse.from_reading(reading, frame).to_dict()

    @app.get("/heatmap")
    def heatmap() -> dict:
        hm = engine.heatmap()
        return {k: v.tolist() for k, v in hm.items()}

    return app
