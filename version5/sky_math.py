"""Vectorised spherical-trigonometry engine (no Python ``for`` loops over points).

Given the ten bodies' absolute equatorial coordinates ``(alpha, delta)`` for one
timestamp and a batch of ``N`` geographic observers, this computes every observer's
local horizon (altitude, azimuth, hour-angle) in a single broadcasted operation of
shape ``[N, 10]``. The same formulas are re-implemented, line for line, in
``version5/web/main.js`` so the browser's math is bit-for-bit the server's math
(PRD page 10, "Visual & Mathematical Validation").

Convention: azimuth follows Meeus (measured from the **south**, positive toward the
**west**), written with an ``atan2`` so it is quadrant-correct and continuous. The
absolute zero-point is irrelevant to the field as long as Python, JS and the trained
model all share this one definition — which they do.

Angles in / out are **radians**. Only NumPy is used (the data workers are CPU-bound
and this keeps them torch-free); the identical broadcast works unchanged in torch.
"""

from __future__ import annotations

import numpy as np

from .config import VIGHATIKA_DAYS

# raw feature layout of the [N,10,5] tensor fed to the encoder
COL_ALT, COL_AZ, COL_RA, COL_DEC, COL_HA = 0, 1, 2, 3, 4

__all__ = [
    "COL_ALT", "COL_AZ", "COL_RA", "COL_DEC", "COL_HA",
    "local_features", "recon_target", "sample_locations", "random_jd_quantized",
]


def _wrap_pi(a: np.ndarray) -> np.ndarray:
    """Wrap an angle to the canonical ``(-pi, pi]`` (matches JS ``atan2``)."""
    return np.arctan2(np.sin(a), np.cos(a))


def local_features(eq: np.ndarray, gast_rad: float,
                   lat_rad: np.ndarray, lon_rad: np.ndarray) -> np.ndarray:
    """Local sky matrix ``[N, 10, 5]`` for ``N`` observers at one timestamp.

    Parameters
    ----------
    eq : ``(10, 4)`` equatorial state ``[ra_deg, dec_deg, dist_au, ra_speed]``
        (from :func:`version5.ephemeris.equatorial_state`).
    gast_rad : Greenwich Apparent Sidereal Time (radians).
    lat_rad, lon_rad : ``(N,)`` observer latitude / longitude (radians).

    Columns of the last axis: ``[altitude, azimuth, RA, declination, hour_angle]``.
    """
    ra = np.deg2rad(eq[:, 0])[None, :]                       # [1,10]
    dec = np.deg2rad(eq[:, 1])[None, :]                      # [1,10]
    phi = np.asarray(lat_rad, dtype=np.float64)[:, None]     # [N,1]
    lam = np.asarray(lon_rad, dtype=np.float64)[:, None]     # [N,1]

    lst = gast_rad + lam                                     # [N,1] local sidereal time
    ha = _wrap_pi(lst - ra)                                  # [N,10] hour angle

    sin_phi, cos_phi = np.sin(phi), np.cos(phi)
    sin_dec, cos_dec = np.sin(dec), np.cos(dec)
    sin_ha, cos_ha = np.sin(ha), np.cos(ha)

    sin_alt = sin_phi * sin_dec + cos_phi * cos_dec * cos_ha
    alt = np.arcsin(np.clip(sin_alt, -1.0, 1.0))            # [N,10]
    az = np.arctan2(sin_ha * cos_dec,
                    cos_ha * sin_phi * cos_dec - sin_dec * cos_phi)   # [N,10]

    n = phi.shape[0]
    feats = np.empty((n, ra.shape[1], 5), dtype=np.float32)
    feats[:, :, COL_ALT] = alt
    feats[:, :, COL_AZ] = az
    feats[:, :, COL_RA] = np.broadcast_to(ra, (n, ra.shape[1]))
    feats[:, :, COL_DEC] = np.broadcast_to(dec, (n, dec.shape[1]))
    feats[:, :, COL_HA] = ha
    return feats


def recon_target(feats: np.ndarray) -> np.ndarray:
    """Reconstruction target ``[N, 10, 4]`` = ``(sin,cos)`` of altitude & azimuth.

    Representing the two local horizon angles by their sine and cosine makes the
    autoencoder's MSE loss wrap-safe (a planet crossing due-south/north azimuth is
    continuous), and proves the 3 OKLab neurons fully describe the local geometry.
    """
    alt = feats[..., COL_ALT]
    az = feats[..., COL_AZ]
    out = np.stack([np.sin(alt), np.cos(alt), np.sin(az), np.cos(az)], axis=-1)
    return out.astype(np.float32)


def sample_locations(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    """``n`` observer locations uniform over the sphere's **area** (radians).

    Longitude is uniform in ``[-pi, pi)``; latitude uses ``arcsin(U(-1,1))`` so the
    poles are not over-sampled (PRD page 2, "The Polar Trap").
    """
    lat = np.arcsin(rng.uniform(-1.0, 1.0, size=n))
    lon = rng.uniform(-np.pi, np.pi, size=n)
    return lat.astype(np.float64), lon.astype(np.float64)


def random_jd_quantized(rng: np.random.Generator,
                        start_jd: float, end_jd: float) -> float:
    """A random Julian Day snapped to the 24-second (Vighatika) grid.

    Uniform over the whole span *and* exactly quantised: we pick a random integer
    number of 24-second ticks from the start, guaranteeing both century-hopping
    coverage and micro-movement sensitivity.
    """
    n_ticks = int((end_jd - start_jd) / VIGHATIKA_DAYS)
    tick = int(rng.integers(0, max(n_ticks, 1)))
    return start_jd + tick * VIGHATIKA_DAYS
