"""Vectorised builder of the Zero-Redundancy 50-D physical state (version5.1).

For a batch of ``N`` geographic observers at one timestamp this assembles, by pure
tensor broadcasting (no Python loop over the points), the strictly non-redundant
``[N, 50]`` state fed to the metric-learning encoder AND to the isometric distance
loss:

* 11 bodies (Sun..Pluto + True Node) x 4 = 44 dims: each body's ecliptic direction as
  a 3D Cartesian **unit vector** ``(X,Y,Z)`` plus its ``tanh``-normalised longitude
  velocity ``V`` — these depend on time only, so they are identical across the batch.
* 2 observer anchors (Ascendant, Midheaven) x 3 = 6 dims: each as a 3D **ecliptic**
  Cartesian unit vector (``beta = 0``) — these depend on the observer's location.

The identical formulas live in ``version5/web/skymath.js`` so the browser's state is
bit-for-bit the server's. NumPy only (the CPU data workers stay torch-free).
"""

from __future__ import annotations

import numpy as np

from kalachakra.local_autoencoder.features import ANG_VEL_SCALE   # peak lunar speed ~15 deg/day

from .config import BODY_FEATURES, N_ML_BODIES, OBS_FEATURES, STATE_DIM, VIGHATIKA_DAYS

#: Indices into the 12-body ecliptic array selecting the 11 ML bodies — the Mean Node
#: (index 10) is dropped as geometrically redundant with the True Node.
ML_BODY_INDICES: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11)
_BODY_DIM = N_ML_BODIES * BODY_FEATURES                         # 44

__all__ = [
    "ML_BODY_INDICES", "STATE_DIM", "ascendant_mc_vertex", "body_state_flat",
    "local_state", "sample_locations", "random_jd_quantized",
]


def ascendant_mc_vertex(ramc: np.ndarray, phi: np.ndarray, eps: float):
    """Vectorised Ascendant, Midheaven and Vertex (ecliptic longitudes, radians).

    Pure spherical trig from the Local Sidereal Time ``ramc`` and observer latitude
    ``phi`` (both broadcastable ``[N,1]``) and obliquity ``eps`` — **not**
    ``swe.houses()``. The ``cos phi`` / ``sin phi`` factored forms stay finite at the
    poles. Asc & MC match pyswisseph's native ``swe.houses`` to ~1e-13 deg.
    """
    st, ct = np.sin(ramc), np.cos(ramc)
    sp, cp = np.sin(phi), np.cos(phi)
    ce, se = np.cos(eps), np.sin(eps)
    mc = np.arctan2(st, ct * ce)
    asc = np.arctan2(ct * cp, -(st * ce * cp + sp * se))
    vx = np.arctan2(ct * sp, -(st * ce * sp + cp * se))
    return asc, mc, vx


def body_state_flat(ecl: np.ndarray) -> np.ndarray:
    """The 44 location-independent body dims ``[X,Y,Z,V] x 11`` for one timestamp.

    ``ecl`` is the ``(12,4)`` ecliptic state ``[lon_deg, lat_deg, dist_au, lon_speed]``.
    """
    e = ecl[ML_BODY_INDICES, :]                                # (11,4)
    lam = np.deg2rad(e[:, 0])
    bet = np.deg2rad(e[:, 1])
    vel = np.tanh(e[:, 3] / ANG_VEL_SCALE)                     # V = tanh(v_raw / v_max)
    cb = np.cos(bet)
    x = cb * np.cos(lam)                                        # ecliptic Cartesian unit vec
    y = cb * np.sin(lam)
    z = np.sin(bet)
    return np.stack([x, y, z, vel], axis=1).reshape(-1).astype(np.float64)   # (44,)


def local_state(ecl: np.ndarray, eps: float, gast_rad: float,
                lat_rad: np.ndarray, lon_rad: np.ndarray) -> np.ndarray:
    """Assemble the ``[N, 50]`` non-redundant physical state for one timestamp."""
    phi = np.asarray(lat_rad, dtype=np.float64)[:, None]       # [N,1]
    lon = np.asarray(lon_rad, dtype=np.float64)[:, None]
    n = phi.shape[0]

    asc, mc, _vx = ascendant_mc_vertex(gast_rad + lon, phi, eps)   # each [N,1]

    state = np.empty((n, STATE_DIM), dtype=np.float32)
    state[:, :_BODY_DIM] = body_state_flat(ecl)[None, :]       # broadcast time-only bodies
    # observer anchors as 3D ecliptic Cartesian unit vectors (beta = 0 -> Z = 0)
    state[:, _BODY_DIM + 0] = np.cos(asc)[:, 0]
    state[:, _BODY_DIM + 1] = np.sin(asc)[:, 0]
    state[:, _BODY_DIM + 2] = 0.0
    state[:, _BODY_DIM + 3] = np.cos(mc)[:, 0]
    state[:, _BODY_DIM + 4] = np.sin(mc)[:, 0]
    state[:, _BODY_DIM + 5] = 0.0
    assert state.shape[1] == STATE_DIM and OBS_FEATURES == 6   # layout guard
    return state


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
