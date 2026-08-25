"""Celestial feature extraction + terrestrial coordinate sampling.

This is the bridge between the existing Swiss-Ephemeris loader and the decoupled
models. For each instant it reads the ten primary bodies (Sun..Pluto) directly
from ``kalachakra.ephemeris.global_state`` and builds the wrap-continuous
``(10, 5)`` celestial tensor. It also samples continuous terrestrial coordinates
uniformly over the sphere's *area* (not uniformly in latitude, which would clump
at the poles) so the Earth Lens is trained on an unbiased spherical measure.

Every angular quantity is represented by its ``(sin, cos)`` projection so there is
no discontinuity at the 0/360-degree seam; the only raw scalar is the longitudinal
angular velocity, kept in radians/day (a natural unit, no arbitrary rescaling).
"""

from __future__ import annotations

import numpy as np
import torch

from ..ephemeris import global_state as gs

#: Swiss-Ephemeris ids for the ten primary bodies (Sun=0 .. Pluto=9).
BODY_SWE_IDS: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
BODY_NAMES: tuple[str, ...] = (
    "Sun", "Moon", "Mercury", "Venus", "Mars",
    "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto",
)
N_BODIES = len(BODY_SWE_IDS)


# ---------------------------------------------------------------------------
# celestial state -> (10, 5) tensor
# ---------------------------------------------------------------------------
def raw_bodies(jd_ut: float):
    """(lon, lat, lon_velocity) in radians / radians / radians-per-day, shape (10,)."""
    gs._require_swe()
    flags = gs._calc_flags()                       # honours the configured backend + speed
    lon = np.empty(N_BODIES, dtype=np.float64)
    lat = np.empty(N_BODIES, dtype=np.float64)
    vel = np.empty(N_BODIES, dtype=np.float64)
    for i, swe_id in enumerate(BODY_SWE_IDS):
        values, _flag = gs.swe.calc_ut(float(jd_ut), swe_id, flags)
        lon_deg, lat_deg, _dist, lon_sp, _lat_sp, _dist_sp = values[:6]
        lon[i] = np.deg2rad(lon_deg % 360.0)
        lat[i] = np.deg2rad(lat_deg)
        vel[i] = np.deg2rad(lon_sp)                # deg/day -> rad/day
    return lon, lat, vel


def encode_celestial(lon: np.ndarray, lat: np.ndarray, vel: np.ndarray) -> np.ndarray:
    """Pack raw angles into the wrap-continuous ``(10, 5)`` feature tensor."""
    return np.stack(
        [np.sin(lon), np.cos(lon), np.sin(lat), np.cos(lat), vel], axis=-1
    ).astype(np.float32)


def celestial_features(jd_ut: float) -> np.ndarray:
    """One instant -> ``(10, 5)`` celestial feature tensor."""
    return encode_celestial(*raw_bodies(jd_ut))


def celestial_features_batch(jds_ut) -> np.ndarray:
    """Many instants -> ``(T, 10, 5)`` celestial feature tensor."""
    jds = np.asarray(jds_ut, dtype=np.float64).ravel()
    return np.stack([celestial_features(float(j)) for j in jds], axis=0)


def decode_lonlat_np(features: np.ndarray):
    """Inverse of :func:`encode_celestial`: recover ``(lon, lat)`` radians (numpy)."""
    f = np.asarray(features)
    lon = np.arctan2(f[..., 0], f[..., 1])
    lat = np.arctan2(f[..., 2], f[..., 3])
    return lon, lat


def decode_lon(features: torch.Tensor) -> torch.Tensor:
    """Recover body ecliptic longitude (radians) from a ``(...,10,5)`` torch tensor."""
    return torch.atan2(features[..., 0], features[..., 1])


# ---------------------------------------------------------------------------
# terrestrial coordinate sampling (continuous, area-uniform on S^2)
# ---------------------------------------------------------------------------
def sample_sphere_coords(n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` random ``(lat, lon)`` radian pairs, uniform over the sphere's area.

    ``lon ~ U(-pi, pi)`` and ``sin(lat) ~ U(-1, 1)`` (so ``lat = arcsin(u)``) gives
    the uniform measure on S^2 -- the correct, heuristic-free spherical prior.
    """
    lon = rng.uniform(-np.pi, np.pi, size=n)
    lat = np.arcsin(rng.uniform(-1.0, 1.0, size=n))
    return np.stack([lat, lon], axis=-1).astype(np.float32)


def equirect_grid(width: int, height: int) -> np.ndarray:
    """A ``(height*width, 2)`` equirectangular grid of ``(lat, lon)`` radians.

    Row-major (lat outer, lon inner) so a reshape to ``(height, width, ...)`` yields
    a north-up equirectangular image ready for a WebGL texture upload.
    """
    lats = np.linspace(np.pi / 2, -np.pi / 2, height, dtype=np.float32)     # +90 -> -90
    lons = np.linspace(-np.pi, np.pi, width, endpoint=False, dtype=np.float32)
    lat_g, lon_g = np.meshgrid(lats, lons, indexing="ij")
    return np.stack([lat_g.ravel(), lon_g.ravel()], axis=-1)


def geodesic_neighbor(latlon: torch.Tensor, eps_rad: float,
                      generator: torch.Generator | None = None) -> torch.Tensor:
    """A point a geodesic distance ``eps_rad`` from each ``(lat, lon)`` on a random
    bearing -- the second sample for a finite-difference spatial gradient.

    Uses the exact great-circle direct formula, so the offset is a true geodesic
    step on the sphere (no flat-plane approximation).
    """
    lat = latlon[..., 0]
    lon = latlon[..., 1]
    theta = torch.rand(lat.shape, generator=generator, device=latlon.device,
                       dtype=latlon.dtype) * (2.0 * np.pi)        # random bearing
    sin_d, cos_d = np.sin(eps_rad), np.cos(eps_rad)
    lat2 = torch.asin(torch.clamp(
        torch.sin(lat) * cos_d + torch.cos(lat) * sin_d * torch.cos(theta),
        -1.0, 1.0))
    lon2 = lon + torch.atan2(
        torch.sin(theta) * sin_d * torch.cos(lat),
        cos_d - torch.sin(lat) * torch.sin(lat2))
    return torch.stack([lat2, lon2], dim=-1)


def latlon_to_unit_vector(latlon: torch.Tensor) -> torch.Tensor:
    """``(..., 2)`` ``(lat, lon)`` radians -> ``(..., 3)`` unit vectors on S^2 (torch).

    This lift is what makes the terrestrial coordinate continuous everywhere: the
    +/-180-degree meridian seam and the pole singularities of a raw ``(lat, lon)``
    parameterisation both vanish on the unit sphere.
    """
    lat = latlon[..., 0]
    lon = latlon[..., 1]
    cos_lat = torch.cos(lat)
    return torch.stack(
        [cos_lat * torch.cos(lon), cos_lat * torch.sin(lon), torch.sin(lat)],
        dim=-1,
    )
