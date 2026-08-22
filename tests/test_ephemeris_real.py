"""Tests that exercise the real ephemeris. Skipped if pyswisseph is absent."""

import numpy as np
import pytest

from kalachakra.ephemeris import global_state
from kalachakra.ephemeris.calendar import parse_datetime

pytestmark = pytest.mark.skipif(
    not global_state.ephemeris_available(),
    reason="pyswisseph not installed",
)


def test_sun_longitude_at_j2000():
    # The Sun sits at ~280.4 deg (early Capricorn, tropical) on 2000-01-01.
    g = global_state.global_state_frame(2451545.0)
    lon = np.rad2deg(np.arctan2(g[0, 1], g[0, 0])) % 360.0
    assert 279.0 < lon < 282.0


def test_moon_moves_fast_saturn_slow():
    g = global_state.global_state_frame(2451545.0)
    moon_speed = abs(np.rad2deg(g[1, 3]))   # deg/day
    saturn_speed = abs(np.rad2deg(g[6, 3]))
    assert moon_speed > 10.0                # Moon ~13 deg/day
    assert saturn_speed < 0.2               # Saturn ~0.03 deg/day


def test_ketu_opposes_rahu():
    g = global_state.global_state_frame(2451545.0)
    rahu = np.rad2deg(np.arctan2(g[7, 1], g[7, 0])) % 360.0
    ketu = np.rad2deg(np.arctan2(g[8, 1], g[8, 0])) % 360.0
    sep = abs(rahu - ketu) % 360.0
    assert abs(sep - 180.0) < 1e-3


def test_real_2024_solar_eclipse_signature():
    from kalachakra.analysis import weather
    jd = parse_datetime("2024-04-08T18:17:00Z")
    sig = weather.frame_signature(jd)
    assert sig.eclipse["is_eclipse"]
    assert sig.eclipse["sun_moon_sep_deg"] < 1.0   # near-exact new moon
