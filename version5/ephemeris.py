"""The *single query* into ``pyswisseph`` — the only place version5 touches the
ephemeris files.

The equatorial coordinates (Right Ascension alpha, Declination delta) of a planet
depend on **time only, not location**. So for any timestamp we query the ephemeris
exactly ten times — once per body — and the resulting ``(10, ...)`` block is the
master reference broadcast over every geographic observer (training) or every screen
pixel (rendering). This is the mathematical redundancy the previous per-coordinate
versions paid for on every point.

Everything here delegates the actual file access to
:mod:`kalachakra.ephemeris.global_state`; nothing is duplicated.
"""

from __future__ import annotations

import numpy as np

from kalachakra.ephemeris import global_state as gs
from kalachakra.local_autoencoder.features import BODY_NAMES, BODY_SWE_IDS

# pyswisseph flag: return apparent equatorial (RA, Dec) instead of ecliptic. The
# value matches ``swisseph.FLG_EQUATORIAL``; stated as a literal so this module
# imports even when the native package is absent (mirrors global_state's style).
_FLG_EQUATORIAL: int = 2048

#: Columns of the equatorial-state block, per body.
EQ_COLS = ("ra_deg", "dec_deg", "dist_au", "ra_speed_deg_per_day")

__all__ = [
    "BODY_NAMES", "EQ_COLS", "configure", "equatorial_state", "gast_hours",
    "gast_radians", "telemetry",
]


def configure(ephe_path: str | None = None, jpl_file: str | None = None) -> str:
    """Select the ephemeris backend (explicit flags win, else auto). Returns mode."""
    if not gs.ephemeris_available():
        raise RuntimeError("pyswisseph is required for version5.")
    return gs.configure_from_args(ephe_path=ephe_path, jpl_file=jpl_file)


def equatorial_state(jd_ut: float) -> np.ndarray:
    """The single query: ``(10, 4)`` apparent equatorial state via ``calc_ut``.

    Columns ``[ra_deg, dec_deg, dist_au, ra_speed_deg_per_day]``. Ten ``calc_ut``
    calls and nothing else — no location, no grid, no loop over observers.
    """
    gs._require_swe()
    flags = gs._calc_flags() | _FLG_EQUATORIAL
    jd = float(jd_ut)
    eq = np.empty((len(BODY_SWE_IDS), 4), dtype=np.float64)
    for i, sid in enumerate(BODY_SWE_IDS):
        v = gs.swe.calc_ut(jd, sid, flags)[0]
        eq[i] = (v[0], v[1], v[2], v[3])          # ra, dec, dist, ra_speed
    return eq


def gast_hours(jd_ut: float) -> float:
    """Greenwich Apparent Sidereal Time in hours (nutation included).

    ``swe.sidtime`` already returns *apparent* sidereal time, which pairs correctly
    with the apparent RA above so that ``H = GAST - RA`` is the true hour angle.
    """
    gs._require_swe()
    return float(gs.swe.sidtime(float(jd_ut)))


def gast_radians(jd_ut: float) -> float:
    """GAST as an angle in radians (``hours * 15 deg * pi/180``)."""
    return gast_hours(jd_ut) * (np.pi / 12.0)


def telemetry(jd_ut: float) -> dict:
    """The micro-payload for one timestamp: GAST + the ten bodies' RA/Dec.

    Pure numbers (degrees) — the exact contract the ``/telemetry`` endpoint serves
    and the browser consumes. Kept here (not in the server) so it is unit-testable
    without FastAPI and reused verbatim by the vectorised math engine.
    """
    eq = equatorial_state(jd_ut)
    gast_h = gast_hours(jd_ut)
    bodies = {
        name: {
            "ra": round(float(eq[i, 0]), 6),
            "dec": round(float(eq[i, 1]), 6),
            "dist": round(float(eq[i, 2]), 8),
            "ra_speed": round(float(eq[i, 3]), 6),
        }
        for i, name in enumerate(BODY_NAMES)
    }
    return {
        "jd": round(float(jd_ut), 6),
        "gast_hours": round(gast_h, 8),
        "gast_deg": round(gast_h * 15.0, 6),
        "bodies": bodies,
    }
