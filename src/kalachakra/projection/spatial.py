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


def project(global_frame: np.ndarray, jd_ut: float, grid: Grid) -> np.ndarray:
    """Project a single G(t) frame onto every observer node.

    Parameters
    ----------
    global_frame : (N_BODIES, 7) array
        One frame of the global state vector G(t).
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
    eps = float(geo.obliquity_of_ecliptic(jd_ut))
    ra, dec = geo.ecliptic_to_equatorial(lon_ecl, lat_ecl, eps)  # (B,), (B,)

    # Local sidereal time per node -> RAMC (radians).
    gmst_deg = float(geo.greenwich_mean_sidereal_time_deg(jd_ut))
    lst_deg = (gmst_deg + np.rad2deg(grid.lon)) % 360.0        # (N,)
    ramc = np.deg2rad(lst_deg)                                  # (N,)

    phi = grid.lat                                              # (N,)

    # Hour angle H = LST - RA, broadcast to (N, B).
    H = ramc[:, None] - ra[None, :]
    sin_phi = np.sin(phi)[:, None]
    cos_phi = np.cos(phi)[:, None]
    sin_dec = np.sin(dec)[None, :]
    cos_dec = np.cos(dec)[None, :]

    sin_h = sin_phi * sin_dec + cos_phi * cos_dec * np.cos(H)
    sin_h = np.clip(sin_h, -1.0, 1.0)
    h = np.arcsin(sin_h)                                        # altitude (N, B)

    # Azimuth from North, increasing eastward.
    theta = np.arctan2(
        -cos_dec * np.sin(H),
        sin_dec * cos_phi - cos_dec * sin_phi * np.cos(H),
    )                                                          # (N, B)

    # Ascendant offset delta_phi = body ecliptic longitude - ascendant.
    asc = ascendant_longitude(ramc, eps, phi)                  # (N,)
    delta_phi = geo.wrap_angle(lon_ecl[None, :] - asc[:, None])  # (N, B)

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
