"""
Backend-neutral geometric primitives (numpy).

These are the low-level, fully-tested operations that both the analytical
spatial-projection engine (§3.1) and the composite geodesic loss (§5.1) build
on. Keeping them here — in one place, in numpy — means the "true" mathematics is
defined exactly once and can be validated without importing torch.

Angle convention: all public functions take and return **radians** unless the
name ends in ``_deg``.
"""

from __future__ import annotations

import numpy as np

TWO_PI = 2.0 * np.pi


def wrap_angle(theta: np.ndarray | float) -> np.ndarray:
    """Wrap angle(s) into the half-open interval [-pi, pi).

    Cyclic quantities (longitudes, phases) are discontinuous at the 0/2pi
    boundary, which is exactly what confuses a network trained on raw degrees
    (blueprint §2.3). Wrapping keeps differences well-defined.
    """
    return (np.asarray(theta, dtype=np.float64) + np.pi) % TWO_PI - np.pi


def angular_separation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest absolute angular gap between two angles (radians), in [0, pi]."""
    return np.abs(wrap_angle(a - b))


def to_unit_vector(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Map (longitude, latitude) in radians to a 3D unit direction vector.

    Returns an array with a trailing axis of size 3:
        [cos(lat)cos(lon), cos(lat)sin(lon), sin(lat)]
    This is the smooth, boundary-free encoding used throughout the pipeline.
    """
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    cos_lat = np.cos(lat)
    return np.stack(
        [cos_lat * np.cos(lon), cos_lat * np.sin(lon), np.sin(lat)],
        axis=-1,
    )


def geodesic_distance(u: np.ndarray, v: np.ndarray, *, eps: float = 1e-7) -> np.ndarray:
    """Great-circle (geodesic) angular distance between unit vectors, in radians.

    This is the numerically-stable, clamped inverse-cosine formulation named in
    the Geodesic Reconstruction Loss (blueprint §5.1). Inputs are expected to be
    (approximately) unit vectors with the direction on the trailing axis.

    The dot product is clamped to [-1+eps, 1-eps] before ``arccos`` so gradients
    stay finite at the poles of the acos domain.
    """
    dot = np.sum(u * v, axis=-1)
    dot = np.clip(dot, -1.0 + eps, 1.0 - eps)
    return np.arccos(dot)


def pairwise_angular_matrix(lons: np.ndarray) -> np.ndarray:
    """Full N x N matrix of angular separations among a set of longitudes.

    Used by the Aspect Relational Invariance Loss (§5.1) to describe the
    multi-body geometry (conjunctions ~0, oppositions ~pi) in a way that is
    invariant to any global rotation of the whole configuration.
    """
    lons = np.asarray(lons, dtype=np.float64)
    diff = lons[..., :, None] - lons[..., None, :]
    return angular_separation(diff, 0.0)


def obliquity_of_ecliptic(jd: np.ndarray | float) -> np.ndarray:
    """Mean obliquity of the ecliptic (radians) for Julian Day ``jd``.

    IAU 1980 mean-obliquity polynomial in Julian centuries from J2000.0. Over the
    10,256-year span this stays accurate to well under an arc-minute, which is
    ample for a topological (not astrometric) simulation.
    """
    T = (np.asarray(jd, dtype=np.float64) - 2_451_545.0) / 36_525.0
    eps_deg = (
        23.439291
        - 0.0130042 * T
        - 1.64e-7 * T * T
        + 5.04e-7 * T * T * T
    )
    return np.deg2rad(eps_deg)


def greenwich_mean_sidereal_time_deg(jd_ut: np.ndarray | float) -> np.ndarray:
    """Greenwich Mean Sidereal Time in degrees for a UT Julian Day.

    IAU 1982 series (Meeus, *Astronomical Algorithms*, ch. 12). Returned in the
    range [0, 360).
    """
    jd_ut = np.asarray(jd_ut, dtype=np.float64)
    d = jd_ut - 2_451_545.0
    T = d / 36_525.0
    gmst = (
        280.46061837
        + 360.98564736629 * d
        + 0.000387933 * T * T
        - (T * T * T) / 38_710_000.0
    )
    return gmst % 360.0


def ecliptic_to_equatorial(lon: np.ndarray, lat: np.ndarray, eps: np.ndarray):
    """Convert ecliptic (lon, lat) to equatorial (right ascension, declination).

    All inputs and outputs are in radians. ``eps`` is the obliquity.
    Right ascension is returned wrapped to [0, 2pi).
    """
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    sin_dec = np.sin(lat) * np.cos(eps) + np.cos(lat) * np.sin(eps) * np.sin(lon)
    dec = np.arcsin(np.clip(sin_dec, -1.0, 1.0))
    ra = np.arctan2(
        np.sin(lon) * np.cos(eps) - np.tan(lat) * np.sin(eps),
        np.cos(lon),
    )
    return ra % TWO_PI, dec
