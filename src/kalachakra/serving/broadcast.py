"""
Cosmic Weather Broadcast Engine (blueprint §7.1).

A spatial query processor that turns the compressed latent metrics back into
continuous geospatial values. It holds, for a chosen epoch, the geometric
potential field and temporal shear gradient over the 122,880-node mesh plus the
active cluster label per node, and answers point queries by nearest geodesic node.

Pure numpy; the network/API surface lives in :mod:`kalachakra.serving.api`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..grid.geodesic import Grid


@dataclass
class LocalReading:
    """The broadcast payload for one geographic point at one frame."""

    latitude_deg: float
    longitude_deg: float
    node_index: int
    potential_index: float
    shear_velocity: float
    cluster_id: int


class BroadcastEngine:
    """Serves per-coordinate topological metrics for a fixed timeline frame."""

    def __init__(self, grid: Grid, potential: np.ndarray, shear: np.ndarray,
                 cluster_labels: np.ndarray | None = None):
        n = grid.n_nodes
        if potential.shape != (n,) or shear.shape != (n,):
            raise ValueError("potential/shear must be shape (N_nodes,)")
        self.grid = grid
        self.potential = np.asarray(potential, dtype=np.float64)
        self.shear = np.asarray(shear, dtype=np.float64)
        self.cluster_labels = (
            np.full(n, -1, dtype=np.int64)
            if cluster_labels is None
            else np.asarray(cluster_labels, dtype=np.int64)
        )

    def nearest_node(self, lat_deg: float, lon_deg: float) -> int:
        """Index of the mesh node closest (great-circle) to the query point."""
        lat = np.deg2rad(lat_deg)
        lon = np.deg2rad(lon_deg)
        q = np.array([np.cos(lat) * np.cos(lon),
                      np.cos(lat) * np.sin(lon),
                      np.sin(lat)])
        return int(np.argmax(self.grid.xyz @ q))

    def query(self, lat_deg: float, lon_deg: float) -> LocalReading:
        idx = self.nearest_node(lat_deg, lon_deg)
        return LocalReading(
            latitude_deg=lat_deg,
            longitude_deg=lon_deg,
            node_index=idx,
            potential_index=float(self.potential[idx]),
            shear_velocity=float(self.shear[idx]),
            cluster_id=int(self.cluster_labels[idx]),
        )

    def heatmap(self) -> dict[str, np.ndarray]:
        """Full-mesh arrays for the WebGL client (lat/lon in degrees)."""
        return {
            "lat_deg": np.rad2deg(self.grid.lat),
            "lon_deg": np.rad2deg(self.grid.lon),
            "potential": self.potential,
            "shear": self.shear,
            "cluster": self.cluster_labels,
        }
