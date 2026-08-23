"""
Analytical spatial projection engine: G(t) -> E(t, s) (blueprint §3.1).

The global state is projected onto every observer node with pure closed-form
spherical trigonometry — no ephemeris query per coordinate. For each body the
engine computes local Azimuth (theta) and Altitude (h) relative to the node's
horizon, plus the Ascendant offset (delta phi = body longitude - rising point),
and encodes them as the boundary-free 5-vector

    e_i(s, t) = [ cos(theta)cos(h), sin(theta)cos(h), sin(h),
                  cos(delta_phi), sin(delta_phi) ]

The reference implementation here is numpy and vectorized over
(nodes x bodies); on device this is the single broadcast tensor multiply the
blueprint executes via Metal Performance Shaders. The math is identical, so this
module doubles as the correctness oracle for the GPU kernel.

Azimuth convention: measured from geographic North, increasing toward East.
"""

from __future__ import annotations

import numpy as np

from .. import constants as C
from .. import geometry as geo
from ..grid.geodesic import Grid


def decode_ecliptic(global_frame: np.ndarray):
    """Recover (longitude, latitude) in radians from the G(t) unit-vector rows.

    ``global_frame`` has shape ``(N_BODIES, 7)``; the first three columns are the
    3D unit direction. Returns ``(lon[B], lat[B])``.
    """
    x, y, z = global_frame[:, 0], global_frame[:, 1], global_frame[:, 2]
    lat = np.arcsin(np.clip(z, -1.0, 1.0))
    lon = np.arctan2(y, x)
    return lon, lat


def ascendant_longitude(ramc_rad: np.ndarray, obliquity: float,
                        geo_lat: np.ndarray) -> np.ndarray:
    """Ecliptic longitude of the Ascendant (rising point), in radians.

    Standard closed form from the Right Ascension of the Meridian (RAMC), the
    obliquity of the ecliptic and the observer's geographic latitude. Returned in
    [0, 2pi). At |geo_lat| approaching the polar circles the Ascendant becomes
    ill-conditioned; the smooth cos/sin encoding downstream keeps the field finite.
    """
    asc = np.arctan2(
        np.cos(ramc_rad),
        -(np.sin(ramc_rad) * np.cos(obliquity)
          + np.tan(geo_lat) * np.sin(obliquity)),
    )
    return asc % geo.TWO_PI


def _ecliptic_to_equatorial_xyz(xyz_ecl: np.ndarray, eps: float) -> np.ndarray:
    """Rotate ecliptic rectangular coords to equatorial (about the x-axis)."""
    x, y, z = xyz_ecl[..., 0], xyz_ecl[..., 1], xyz_ecl[..., 2]
    ce, se = np.cos(eps), np.sin(eps)
    return np.stack([x, y * ce - z * se, y * se + z * ce], axis=-1)


def _equatorial_to_ecliptic_xyz(xyz_eq: np.ndarray, eps: float) -> np.ndarray:
    """Rotate equatorial rectangular coords back to ecliptic (about the x-axis)."""
    x, y, z = xyz_eq[..., 0], xyz_eq[..., 1], xyz_eq[..., 2]
    ce, se = np.cos(eps), np.sin(eps)
    return np.stack([x, y * ce + z * se, -y * se + z * ce], axis=-1)


def project(global_frame: np.ndarray, jd_ut: float, grid: Grid) -> np.ndarray:
    """Project a single G(t) frame onto every observer node (topocentric).

    The field is **topocentric**: each observer sees each body from Earth's
    surface, not from the geocentre. The geocentric direction and distance carried
    in ``global_frame`` (columns 0-2 and 5) are combined with the observer's
    surface position to form the true local direction, so lunar parallax (~0.95
    deg between opposite limbs of the Earth) is resolved — which is what makes an
    eclipse localize to its ground track rather than glowing identically for the
    whole hemisphere. This stays a pure broadcast of G(t) onto the mesh: no
    per-node ephemeris query, preserving the global/local decoupling (§3.1).

    Parameters
    ----------
    global_frame : (N_BODIES, 7) array
        One frame of G(t): columns 0-2 are the geocentric ecliptic unit
        direction, column 5 is the geocentric distance in AU.
    jd_ut : float
        Julian Day (UT) of the frame — sets sidereal time and obliquity.
    grid : Grid
        Observer mesh (provides per-node geographic lat/lon).

    Returns
    -------
    (N_nodes, N_BODIES, 5) array
        The local topocentric field E(t, s).
    """
    lon_ecl, lat_ecl = decode_ecliptic(global_frame)          # (B,)
    dist = np.asarray(global_frame[:, 5], dtype=np.float64)    # (B,) AU (geocentric)
    eps = float(geo.obliquity_of_ecliptic(jd_ut))

    # Geocentric body position vectors in equatorial rectangular AU.
    unit_ecl = global_frame[:, :3].astype(np.float64)         # (B, 3) ecliptic unit
    r_body_ecl = dist[:, None] * unit_ecl                      # (B, 3) AU
    r_body_eq = _ecliptic_to_equatorial_xyz(r_body_ecl, eps)  # (B, 3) AU

    # Local sidereal time per node -> RAMC / observer position (radians).
    gmst_deg = float(geo.greenwich_mean_sidereal_time_deg(jd_ut))
    lst_deg = (gmst_deg + np.rad2deg(grid.lon)) % 360.0        # (N,)
    ramc = np.deg2rad(lst_deg)                                  # (N,)
    phi = grid.lat                                              # (N,)

    # Observer's geocentre-to-surface vector, equatorial rectangular AU (spherical
    # Earth). This is the term that makes the field topocentric.
    rho = C.EARTH_RADIUS_AU
    obs_eq = rho * np.stack(
        [np.cos(phi) * np.cos(ramc), np.cos(phi) * np.sin(ramc), np.sin(phi)],
        axis=-1,
    )                                                          # (N, 3) AU

    # Topocentric body vectors: geocentric body minus observer, per (node, body).
    r_topo_eq = r_body_eq[None, :, :] - obs_eq[:, None, :]     # (N, B, 3) AU
    x, y, z = r_topo_eq[..., 0], r_topo_eq[..., 1], r_topo_eq[..., 2]
    r_topo = np.sqrt(x * x + y * y + z * z)                    # (N, B)
    ra = np.arctan2(y, x)                                       # topo RA (N, B)
    dec = np.arcsin(np.clip(z / r_topo, -1.0, 1.0))            # topo dec (N, B)

    # Hour angle H = LST - RA (both now node-dependent), altitude and azimuth.
    H = ramc[:, None] - ra
    sin_phi = np.sin(phi)[:, None]
    cos_phi = np.cos(phi)[:, None]
    sin_dec = np.sin(dec)
    cos_dec = np.cos(dec)

    sin_h = sin_phi * sin_dec + cos_phi * cos_dec * np.cos(H)
    sin_h = np.clip(sin_h, -1.0, 1.0)
    h = np.arcsin(sin_h)                                        # altitude (N, B)

    theta = np.arctan2(
        -cos_dec * np.sin(H),
        sin_dec * cos_phi - cos_dec * sin_phi * np.cos(H),
    )                                                          # azimuth (N, B)

    # Topocentric ecliptic longitude for the Ascendant offset term.
    r_topo_ecl = _equatorial_to_ecliptic_xyz(r_topo_eq, eps)   # (N, B, 3)
    lon_topo = np.arctan2(r_topo_ecl[..., 1], r_topo_ecl[..., 0])  # (N, B)
    asc = ascendant_longitude(ramc, eps, phi)                  # (N,)
    delta_phi = geo.wrap_angle(lon_topo - asc[:, None])        # (N, B)

    cos_h = np.cos(h)
    field = np.stack(
        [
            np.cos(theta) * cos_h,
            np.sin(theta) * cos_h,
            np.sin(h),
            np.cos(delta_phi),
            np.sin(delta_phi),
        ],
        axis=-1,
    )                                                          # (N, B, 5)
    assert field.shape == (grid.n_nodes, C.N_BODIES, C.LOCAL_BODY_FEATURES)
    return field
