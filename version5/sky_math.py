"""Vectorised spherical-trigonometry engine (no Python ``for`` loops over points).

Given the twelve bodies' absolute coordinates for one timestamp and a batch of ``N``
geographic observers, this computes — in a single broadcasted pass of shape
``[N, 12]`` — every observer's local horizon (altitude, azimuth), the high-frequency
geographic resolvers (Ascendant, Midheaven, Vertex), and each body's House offset
relative to the Ascendant. The batch dimension is pure tensor broadcasting so the
~80k samples/s throughput is preserved; the identical formulas are re-implemented in
``version5/web/skymath.js`` so the browser's maths is bit-for-bit the server's.

Angles in / out are **radians**. NumPy only (the CPU data workers stay torch-free);
every operation is a broadcast that maps 1:1 to ``torch`` if run on the GPU.
"""

from __future__ import annotations

import numpy as np

from kalachakra.local_autoencoder.features import ANG_VEL_SCALE   # reuse velocity scale

from .config import VIGHATIKA_DAYS

# raw feature layout of the [N,12,6] body tensor fed to the encoder
COL_ALT, COL_AZ, COL_LON, COL_LAT, COL_HPOS, COL_VEL = 0, 1, 2, 3, 4, 5
#: body columns that are cyclic angles (expanded to sin/cos by the encoder)
ANGULAR_COLS: tuple[int, ...] = (COL_ALT, COL_AZ, COL_LON, COL_LAT, COL_HPOS)
SCALAR_COLS: tuple[int, ...] = (COL_VEL,)

__all__ = [
    "COL_ALT", "COL_AZ", "COL_LON", "COL_LAT", "COL_HPOS", "COL_VEL",
    "ANGULAR_COLS", "SCALAR_COLS", "ascendant_mc_vertex", "local_features",
    "recon_target", "sample_locations", "random_jd_quantized",
]


def _wrap_pi(a: np.ndarray) -> np.ndarray:
    """Wrap an angle to the canonical ``(-pi, pi]`` (matches JS ``atan2``)."""
    return np.arctan2(np.sin(a), np.cos(a))


def ascendant_mc_vertex(ramc: np.ndarray, phi: np.ndarray, eps: float):
    """Vectorised Ascendant, Midheaven and Vertex (ecliptic longitudes, radians).

    Pure spherical trig from the Local Sidereal Time ``ramc`` (= RA of the meridian)
    and observer latitude ``phi`` (both broadcastable ``[N,1]``) and obliquity
    ``eps`` — **not** ``swe.houses()``, so 2,048 observers cost one broadcast, not
    2,048 C calls. The ``cos phi`` / ``sin phi`` factored forms stay finite at the
    poles (no ``tan phi`` blow-up).

    * MC   = atan2(sin θ, cos θ cos ε)
    * Asc  = atan2(cos θ cos φ, −(sin θ cos ε cos φ + sin φ sin ε))
    * Vx   = the co-latitude Ascendant (φ → 90°−φ), the ecliptic point on the prime
             vertical — a smooth latitude-sensitive spatial anchor.
    """
    st, ct = np.sin(ramc), np.cos(ramc)
    sp, cp = np.sin(phi), np.cos(phi)
    ce, se = np.cos(eps), np.sin(eps)
    mc = np.arctan2(st, ct * ce)
    asc = np.arctan2(ct * cp, -(st * ce * cp + sp * se))
    vx = np.arctan2(ct * sp, -(st * ce * sp + cp * se))
    return asc, mc, vx


def local_features(ecl: np.ndarray, eq: np.ndarray, eps: float, gast_rad: float,
                   lat_rad: np.ndarray, lon_rad: np.ndarray):
    """Local sky matrix ``[N,12,6]`` + observer anchors ``[N,3]`` for one timestamp.

    Parameters
    ----------
    ecl : ``(12,4)`` ecliptic state ``[lon_deg, lat_deg, dist_au, lon_speed]``.
    eq  : ``(12,2)`` equatorial ``[ra_deg, dec_deg]`` (from ``ecl_to_equatorial``).
    eps : true obliquity (radians).
    gast_rad : Greenwich Apparent Sidereal Time (radians).
    lat_rad, lon_rad : ``(N,)`` observer latitude / longitude (radians).

    Body columns: ``[altitude, azimuth, ecl_longitude, ecl_latitude, house_offset,
    velocity]``. Observer anchors: ``[Ascendant, Midheaven, Vertex]``.
    """
    ra = np.deg2rad(eq[:, 0])[None, :]                       # [1,12]
    dec = np.deg2rad(eq[:, 1])[None, :]
    lam = np.deg2rad(ecl[:, 0])[None, :]
    bet = np.deg2rad(ecl[:, 1])[None, :]
    vel = (ecl[:, 3] / ANG_VEL_SCALE)[None, :]               # normalised scalar [1,12]
    phi = np.asarray(lat_rad, dtype=np.float64)[:, None]     # [N,1]
    lon = np.asarray(lon_rad, dtype=np.float64)[:, None]

    ramc = gast_rad + lon                                    # local sidereal time [N,1]
    ha = _wrap_pi(ramc - ra)                                 # hour angle [N,12]

    sin_phi, cos_phi = np.sin(phi), np.cos(phi)
    sin_dec, cos_dec = np.sin(dec), np.cos(dec)
    sin_ha, cos_ha = np.sin(ha), np.cos(ha)
    alt = np.arcsin(np.clip(sin_phi * sin_dec + cos_phi * cos_dec * cos_ha, -1.0, 1.0))
    az = np.arctan2(sin_ha * cos_dec,
                    cos_ha * sin_phi * cos_dec - sin_dec * cos_phi)      # [N,12]

    asc, mc, vx = ascendant_mc_vertex(ramc, phi, eps)        # each [N,1]
    hpos = _wrap_pi(lam - asc)                               # house offset [N,12]

    n = phi.shape[0]
    feats = np.empty((n, ra.shape[1], 6), dtype=np.float32)
    feats[:, :, COL_ALT] = alt
    feats[:, :, COL_AZ] = az
    feats[:, :, COL_LON] = np.broadcast_to(lam, (n, lam.shape[1]))
    feats[:, :, COL_LAT] = np.broadcast_to(bet, (n, bet.shape[1]))
    feats[:, :, COL_HPOS] = hpos
    feats[:, :, COL_VEL] = np.broadcast_to(vel, (n, vel.shape[1]))
    obs = np.concatenate([asc, mc, vx], axis=1).astype(np.float32)       # [N,3]
    return feats, obs


def recon_target(feats: np.ndarray) -> np.ndarray:
    """Reconstruction target ``[N,12,4]`` = ``(sin,cos)`` of altitude & azimuth.

    Representing the two local horizon angles by their sine and cosine makes the
    autoencoder's MSE loss wrap-safe and proves the 3 OKLab neurons fully describe
    the local geometry.
    """
    alt = feats[..., COL_ALT]
    az = feats[..., COL_AZ]
    out = np.stack([np.sin(alt), np.cos(alt), np.sin(az), np.cos(az)], axis=-1)
    return out.astype(np.float32)


def sample_locations(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    """``n`` observer locations uniform over the sphere's **area** (radians).

    Longitude uniform in ``[-pi, pi)``; latitude via ``arcsin(U(-1,1))`` so the poles
    are not over-sampled (PRD "The Polar Trap").
    """
    lat = np.arcsin(rng.uniform(-1.0, 1.0, size=n))
    lon = rng.uniform(-np.pi, np.pi, size=n)
    return lat.astype(np.float64), lon.astype(np.float64)


def random_jd_quantized(rng: np.random.Generator,
                        start_jd: float, end_jd: float) -> float:
    """A random Julian Day snapped to the 24-second (Vighatika) grid — uniform over
    the whole span and exactly quantised (micro-movement sensitivity)."""
    n_ticks = int((end_jd - start_jd) / VIGHATIKA_DAYS)
    tick = int(rng.integers(0, max(n_ticks, 1)))
    return start_jd + tick * VIGHATIKA_DAYS
