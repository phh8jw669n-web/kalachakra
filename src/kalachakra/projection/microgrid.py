"""
On-the-fly regional micro-grids for the dynamic LOD engine (blueprint §4).

The continuous analytical projection (`spatial.project`) evaluates *any* geographic
coordinate, so localized "zoom" is exact, not interpolated: given a bounding box
and a density, generate an evenly spaced lat/lng lattice, map it onto the unit
sphere, and hand it to the projection / weather engines. There is no dependence on
the static 122,880-node global mesh.

Pure numpy.
"""

from __future__ import annotations

import numpy as np

from ..grid.geodesic import Grid


def bbox_microgrid(min_lat: float, min_lng: float, max_lat: float, max_lng: float,
                   density: int = 64) -> Grid:
    """Evenly spaced ``density x density`` lat/lng micro-grid over a bounding box.

    ``density`` is the number of samples per axis (so ``density**2`` nodes). The
    grid is mapped onto the unit sphere so the projection engine receives true
    geographic points, with no planar distortion in the physics.
    """
    if density < 2:
        raise ValueError("density must be >= 2")
    if not (min_lat < max_lat and min_lng < max_lng):
        raise ValueError("require min < max for both lat and lng")

    lats = np.linspace(min_lat, max_lat, density)
    lngs = np.linspace(min_lng, max_lng, density)
    grid_lat, grid_lng = np.meshgrid(lats, lngs, indexing="ij")
    lat = np.deg2rad(grid_lat.ravel())
    lon = np.deg2rad(grid_lng.ravel())
    cos_lat = np.cos(lat)
    xyz = np.stack([cos_lat * np.cos(lon), cos_lat * np.sin(lon), np.sin(lat)], axis=1)
    return Grid(xyz=xyz, lat=lat, lon=lon)


def resolution_km(min_lat: float, min_lng: float, max_lat: float, max_lng: float,
                  density: int) -> float:
    """Approximate spatial sample spacing in km (great-circle, mean Earth radius)."""
    r_km = 6371.0
    span_lat = np.deg2rad(max_lat - min_lat)
    span_lng = np.deg2rad(max_lng - min_lng) * np.cos(np.deg2rad((min_lat + max_lat) / 2))
    diag = r_km * np.hypot(span_lat, span_lng)
    return float(diag / max(density - 1, 1))
