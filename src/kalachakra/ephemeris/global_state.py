"""
Global state vector G(t) in R^{10 x 7} (blueprint §2.3).

For each of the ten entities the instantaneous ecliptic state
(lambda, beta, r) and its rates (lambda_dot, beta_dot, r_dot) are read from the
Swiss Ephemeris (DE441) and parameterized into a smooth, boundary-free
seven-dimensional feature vector:

    v_i(t) = [ cos(lam)cos(bet), sin(lam)cos(bet), sin(bet),
               lam_dot, bet_dot, r, r_dot ]

Stacking the ten rows yields G(t). ``pyswisseph`` is imported lazily so this
module can be imported (and unit-tested for shape/encoding) in environments
where the ephemeris is not installed.
"""

from __future__ import annotations

import numpy as np

from .. import constants as C
from . import bodies
from .bodies import Kind

try:  # pragma: no cover - depends on optional native dependency
    import swisseph as swe

    _HAS_SWE = True
except Exception:  # noqa: BLE001 - any import failure means "not available"
    swe = None  # type: ignore[assignment]
    _HAS_SWE = False


# swisseph calc flags: Swiss ephemeris + speed. Declared as literals so the
# module still imports without the native package (values match pyswisseph).
_FLG_SWIEPH = 2
_FLG_SPEED = 256
_CALC_FLAGS = _FLG_SWIEPH | _FLG_SPEED


def ephemeris_available() -> bool:
    """True when pyswisseph could be imported."""
    return _HAS_SWE


def _require_swe() -> None:
    if not _HAS_SWE:
        raise RuntimeError(
            "pyswisseph is not installed. Install `pyswisseph` and point it at "
            "the DE441 data files (see docs/02_global_state.md) to generate G(t)."
        )


def encode_body(lam: float, bet: float, r: float,
                lam_dot: float, bet_dot: float, r_dot: float) -> np.ndarray:
    """Pack one body's raw ecliptic state into its 7-D feature vector.

    ``lam`` and ``bet`` are radians; ``r`` is in AU; the dots are per-day rates
    (radians/day and AU/day). This function is pure and fully testable.
    """
    cos_b = np.cos(bet)
    return np.array(
        [
            np.cos(lam) * cos_b,
            np.sin(lam) * cos_b,
            np.sin(bet),
            lam_dot,
            bet_dot,
            r,
            r_dot,
        ],
        dtype=np.float64,
    )


def _raw_state(entity: bodies.CelestialEntity, jd_ut: float):
    """Return (lam, bet, r, lam_dot, bet_dot, r_dot) in radians / AU / per-day."""
    if entity.kind is Kind.PRECESSION:
        # Ayanamsha: a single precession angle. Encoded as a point on the
        # ecliptic circle (beta = 0, r = 1) whose longitude is psi_t and whose
        # longitude rate is the instantaneous precession rate.
        psi_deg = swe.get_ayanamsa_ut(jd_ut)  # type: ignore[union-attr]
        dt = 1.0  # one day, for a finite-difference precession rate
        psi_deg_next = swe.get_ayanamsa_ut(jd_ut + dt)  # type: ignore[union-attr]
        lam = np.deg2rad(psi_deg)
        lam_dot = np.deg2rad(psi_deg_next - psi_deg) / dt
        return lam, 0.0, 1.0, lam_dot, 0.0, 0.0

    # BODY or NODE: query pyswisseph. Returns (lon, lat, dist, lon_sp, lat_sp, dist_sp).
    values, _flag = swe.calc_ut(jd_ut, entity.swe_id, _CALC_FLAGS)  # type: ignore[union-attr]
    lon_deg, lat_deg, dist_au, lon_sp, lat_sp, dist_sp = values[:6]
    lon_deg += entity.longitude_offset_deg  # Ketu = node + 180 deg
    return (
        np.deg2rad(lon_deg % 360.0),
        np.deg2rad(lat_deg),
        float(dist_au),
        np.deg2rad(lon_sp),   # deg/day -> rad/day
        np.deg2rad(lat_sp),
        float(dist_sp),
    )


def global_state_frame(jd_ut: float) -> np.ndarray:
    """Compute G(t) for a single Julian Day, shape ``(N_BODIES, 7)``."""
    _require_swe()
    rows = np.empty((C.N_BODIES, C.GLOBAL_BODY_FEATURES), dtype=np.float64)
    for i, entity in enumerate(bodies.ENTITIES):
        rows[i] = encode_body(*_raw_state(entity, jd_ut))
    return rows


def global_state_batch(jds_ut: np.ndarray) -> np.ndarray:
    """Compute G(t) for many Julian Days, shape ``(len(jds), N_BODIES, 7)``.

    pyswisseph is scalar, so this loops over frames on the CPU — exactly the
    Phase-1 generation step (§1.3) whose output is serialized to the binary
    store. The heavy lifting downstream (projection, training) is vectorized.
    """
    _require_swe()
    jds_ut = np.asarray(jds_ut, dtype=np.float64)
    out = np.empty((jds_ut.shape[0], C.N_BODIES, C.GLOBAL_BODY_FEATURES),
                   dtype=np.float64)
    for k, jd in enumerate(jds_ut):
        out[k] = global_state_frame(float(jd))
    return out
