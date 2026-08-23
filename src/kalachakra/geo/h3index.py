"""
Uber H3 hexagonal geospatial indexing (blueprint §3).

Every geodesic node is mapped to a 64-bit H3 cell id at resolution 4 (~11,000
km2 cells), with resolution 7 available for localized bounding boxes. Integer
cell ids turn spatial containment and neighborhood lookups into constant-time
set/bitwise operations instead of spherical-trig distance evaluations, which is
what makes the DuckDB spatial filtering fast.

``h3`` (v4) is optional; when absent, a coarse lat/lng bucketing fallback keeps
the API usable for tests, clearly flagged via :func:`h3_available`.
"""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - optional dependency
    import h3

    _HAS_H3 = True
except Exception:  # noqa: BLE001
    h3 = None
    _HAS_H3 = False

BASE_RESOLUTION = 4      # ~11,000 km^2 cells for the global mesh
LOCAL_RESOLUTION = 7     # localized bounding boxes


def h3_available() -> bool:
    return _HAS_H3


def cell_for(lat_deg: float, lng_deg: float, resolution: int = BASE_RESOLUTION) -> int:
    """Integer H3 cell id enclosing the point (h3 v4)."""
    if _HAS_H3:
        return int(h3.str_to_int(h3.latlng_to_cell(lat_deg, lng_deg, resolution)))
    return _fallback_cell(lat_deg, lng_deg, resolution)


def cells_for_grid(lat_deg: np.ndarray, lng_deg: np.ndarray,
                   resolution: int = BASE_RESOLUTION) -> np.ndarray:
    """Vectorized cell ids for arrays of geographic coordinates (int64)."""
    lat = np.asarray(lat_deg, dtype=np.float64).ravel()
    lng = np.asarray(lng_deg, dtype=np.float64).ravel()
    out = np.empty(lat.shape, dtype=np.int64)
    for i in range(lat.shape[0]):
        out[i] = cell_for(float(lat[i]), float(lng[i]), resolution)
    return out


def parent(cell_int: int, resolution: int = BASE_RESOLUTION) -> int:
    """Coarser ancestor cell id at ``resolution``."""
    if _HAS_H3:
        s = h3.int_to_str(int(cell_int))
        return int(h3.str_to_int(h3.cell_to_parent(s, resolution)))
    return int(cell_int)  # fallback ids are not hierarchical


def cells_in_bbox(min_lat: float, min_lng: float, max_lat: float, max_lng: float,
                  resolution: int = BASE_RESOLUTION) -> list[int]:
    """All H3 cell ids covering a lat/lng bounding box (int64)."""
    if _HAS_H3:
        poly = h3.LatLngPoly([
            (min_lat, min_lng), (min_lat, max_lng),
            (max_lat, max_lng), (max_lat, min_lng),
        ])
        return [int(h3.str_to_int(c)) for c in h3.h3shape_to_cells(poly, resolution)]
    # Fallback: enumerate the bucket grid across the box.
    out = set()
    for la in np.linspace(min_lat, max_lat, 16):
        for lo in np.linspace(min_lng, max_lng, 16):
            out.add(_fallback_cell(la, lo, resolution))
    return sorted(out)


def neighbors(cell_int: int, k: int = 1) -> list[int]:
    """Cell ids within grid distance ``k`` (h3 grid_disk)."""
    if _HAS_H3:
        s = h3.int_to_str(int(cell_int))
        return [int(h3.str_to_int(c)) for c in h3.grid_disk(s, k)]
    return [int(cell_int)]


def _fallback_cell(lat_deg: float, lng_deg: float, resolution: int) -> int:
    """Deterministic non-H3 bucket id (coarse; only for h3-less environments)."""
    step = max(0.5, 20.0 / (resolution + 1))
    la = int((lat_deg + 90.0) // step)
    lo = int((lng_deg + 180.0) // step)
    return int(la * 100_000 + lo)
