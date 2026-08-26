"""The *single query* into ``pyswisseph`` — the only place version5 touches the
ephemeris files.

A body's ecliptic state ``(lambda, beta, r, lambda_dot)`` and its equatorial
coordinates ``(alpha, delta)`` depend on **time only, not location**. So for any
timestamp we query the ephemeris exactly twelve times — once per body — and derive
the equatorial coordinates from the ecliptic ones by a single obliquity rotation
(matched to pyswisseph's native equatorial output to ~1e-13 deg). The resulting
``(12, ...)`` block is the master reference broadcast over every geographic observer
(training) or every screen pixel (rendering).

12 bodies: Sun..Pluto (ids 0-9) plus the Mean Node (10) and True Node (11). The query
uses ``FLG_SPEED`` for accurate longitude velocity and the configured backend (Swiss
``.se1`` for the full BCE->CE span, Moshier otherwise).

File access is delegated to :mod:`kalachakra.ephemeris.global_state`; nothing is
duplicated.
"""

from __future__ import annotations

import numpy as np

from kalachakra.ephemeris import global_state as gs
from kalachakra.local_autoencoder.features import BODY_NAMES as _CORE_NAMES
from kalachakra.local_autoencoder.features import BODY_SWE_IDS as _CORE_IDS

# 12 bodies = the ten primaries + the two lunar nodes.
BODY_SWE_IDS: tuple[int, ...] = (*_CORE_IDS, 10, 11)          # + MEAN_NODE, TRUE_NODE
BODY_NAMES: tuple[str, ...] = (*_CORE_NAMES, "MeanNode", "TrueNode")
N_BODIES: int = len(BODY_SWE_IDS)

# pyswisseph special "body" id for the obliquity/nutation of date (== swe.ECL_NUT).
_ECL_NUT: int = -1

#: Columns of the ecliptic-state block, per body.
ECL_COLS = ("lon_deg", "lat_deg", "dist_au", "lon_speed_deg_per_day")

__all__ = [
    "BODY_NAMES", "BODY_SWE_IDS", "N_BODIES", "ECL_COLS", "configure",
    "ecliptic_state", "obliquity_rad", "ecl_to_equatorial", "gast_hours",
    "gast_radians", "telemetry",
]


def configure(ephe_path: str | None = None, jpl_file: str | None = None) -> str:
    """Select the ephemeris backend (explicit flags win, else auto). Returns mode."""
    if not gs.ephemeris_available():
        raise RuntimeError("pyswisseph is required for version5.")
    return gs.configure_from_args(ephe_path=ephe_path, jpl_file=jpl_file)


def ecliptic_state(jd_ut: float) -> np.ndarray:
    """The single query: ``(12, 4)`` apparent ecliptic state via ``calc_ut``.

    Columns ``[lon_deg, lat_deg, dist_au, lon_speed_deg_per_day]`` — the last from
    ``FLG_SPEED``. Twelve ``calc_ut`` calls and nothing else (no location, no loop
    over observers).
    """
    gs._require_swe()
    flags = gs._calc_flags()                                  # backend | FLG_SPEED
    jd = float(jd_ut)
    ecl = np.empty((N_BODIES, 4), dtype=np.float64)
    for i, sid in enumerate(BODY_SWE_IDS):
        v = gs.swe.calc_ut(jd, sid, flags)[0]
        ecl[i] = (v[0], v[1], v[2], v[3])                    # lon, lat, dist, lon_speed
    return ecl


def obliquity_rad(jd_ut: float) -> float:
    """True obliquity of the ecliptic (radians) for the timestamp.

    Using the true obliquity of date (not a fixed 23.439 deg) keeps the derived
    equatorial coordinates and the Ascendant/MC maths exact; it is shipped in the
    telemetry so the browser uses the identical value.
    """
    gs._require_swe()
    return float(np.deg2rad(gs.swe.calc_ut(float(jd_ut), _ECL_NUT, 0)[0][0]))


def ecl_to_equatorial(ecl: np.ndarray, eps: float) -> np.ndarray:
    """Vectorised ecliptic -> equatorial: ``(12,4)`` state -> ``(12,2)`` ``[ra_deg, dec_deg]``.

    Pure tensor rotation over the 12 bodies (no per-body Python astronomy call);
    matches pyswisseph's native ``FLG_EQUATORIAL`` output to ~1e-13 deg.
    """
    lam = np.deg2rad(ecl[:, 0])
    bet = np.deg2rad(ecl[:, 1])
    sin_dec = np.sin(bet) * np.cos(eps) + np.cos(bet) * np.sin(eps) * np.sin(lam)
    dec = np.arcsin(np.clip(sin_dec, -1.0, 1.0))
    ra = np.arctan2(np.sin(lam) * np.cos(eps) - np.tan(bet) * np.sin(eps), np.cos(lam))
    out = np.empty((ecl.shape[0], 2), dtype=np.float64)
    out[:, 0] = np.rad2deg(ra) % 360.0
    out[:, 1] = np.rad2deg(dec)
    return out


def gast_hours(jd_ut: float) -> float:
    """Greenwich Apparent Sidereal Time in hours (nutation included)."""
    gs._require_swe()
    return float(gs.swe.sidtime(float(jd_ut)))


def gast_radians(jd_ut: float) -> float:
    """GAST as an angle in radians (``hours * 15 deg * pi/180``)."""
    return gast_hours(jd_ut) * (np.pi / 12.0)


def telemetry(jd_ut: float) -> dict:
    """The micro-payload for one timestamp: GAST, obliquity, and the twelve bodies'
    equatorial + ecliptic coordinates and longitude velocity.

    Pure numbers (degrees, deg/day) — the exact contract the ``/telemetry`` endpoint
    serves and the browser consumes. Kept here (not in the server) so it is unit
    testable without FastAPI and reused verbatim by the vectorised math engine.
    """
    ecl = ecliptic_state(jd_ut)
    eps = obliquity_rad(jd_ut)
    eq = ecl_to_equatorial(ecl, eps)
    gast_h = gast_hours(jd_ut)
    bodies = {
        name: {
            "ra": round(float(eq[i, 0]), 6),
            "dec": round(float(eq[i, 1]), 6),
            "lon": round(float(ecl[i, 0]), 6),          # ecliptic longitude lambda
            "lat": round(float(ecl[i, 1]), 6),          # ecliptic latitude beta
            "dist": round(float(ecl[i, 2]), 8),
            "lon_speed": round(float(ecl[i, 3]), 6),    # velocity v (deg/day)
        }
        for i, name in enumerate(BODY_NAMES)
    }
    return {
        "jd": round(float(jd_ut), 6),
        "gast_hours": round(gast_h, 8),
        "gast_deg": round(gast_h * 15.0, 6),
        "obliquity_deg": round(float(np.rad2deg(eps)), 6),
        "bodies": bodies,
    }
