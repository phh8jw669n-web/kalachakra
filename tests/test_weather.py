"""Pure-geometry tests for the cosmic-weather engine (no ephemeris needed)."""

import numpy as np

from kalachakra.analysis import weather


def _lons(deg_by_index):
    """Build a length-10 longitude array (radians) from {index: degrees}."""
    a = np.zeros(10)
    for i, d in deg_by_index.items():
        a[i] = np.deg2rad(d)
    return a


def test_separations_fold_to_180():
    lons = _lons({0: 0.0, 1: 200.0})
    sep = weather.separations_deg(lons)
    assert abs(sep[0, 1] - 160.0) < 1e-6  # 200 folds to 160


def test_trine_is_constructive_square_is_destructive():
    w = np.zeros(10); w[0] = w[5] = 1.0  # isolate bodies 0 and 5

    trine = weather.aspect_field(_lons({0: 0.0, 5: 120.0}), weights=w)
    assert trine["resonance"] > 0.9 and trine["tension"] < 0.1

    square = weather.aspect_field(_lons({0: 0.0, 5: 90.0}), weights=w)
    assert square["tension"] > 0.9 and square["resonance"] < 0.1


def test_conjunction_is_constructive():
    w = np.zeros(10); w[0] = w[5] = 1.0
    conj = weather.aspect_field(_lons({0: 0.0, 5: 2.0}), weights=w)
    assert conj["resonance"] > 0.8


def test_stellium_concentration_extremes():
    R_tight, _ = weather.stellium_concentration(np.zeros(10), np.ones(10))
    assert R_tight > 0.99  # all bodies at same longitude

    spread_lons = np.deg2rad(np.linspace(0, 360, 10, endpoint=False))
    R_spread, _ = weather.stellium_concentration(spread_lons, np.ones(10))
    assert R_spread < 0.2  # evenly spread


def test_solar_eclipse_detection():
    # Sun, Moon conjunct at a node -> solar eclipse.
    lons = np.zeros(10)  # Sun(0)=Moon(1)=Rahu(7)=0 deg
    ecl = weather.eclipse_state(lons)
    assert ecl["is_eclipse"] and ecl["solar_proximity"] > 0.5


def test_lunar_eclipse_detection():
    lons = _lons({0: 0.0, 1: 180.0, 7: 0.0})  # Sun 0, Moon 180, node 0
    ecl = weather.eclipse_state(lons)
    assert ecl["lunar_proximity"] > 0.5


def test_no_eclipse_off_node():
    lons = _lons({0: 0.0, 1: 0.0, 7: 90.0})  # Sun/Moon conjunct but off-node
    ecl = weather.eclipse_state(lons)
    assert not ecl["is_eclipse"]


def test_dominant_aspects_sorted_and_labeled():
    lons = _lons({0: 0.0, 5: 120.0, 6: 90.0})
    asp = weather.dominant_aspects(lons)
    assert asp and asp[0]["strength"] >= asp[-1]["strength"]
    assert all(a["kind"] in ("constructive", "destructive") for a in asp)


def test_local_intensity_shape_and_sign():
    rng = np.random.default_rng(0)
    n, b = 50, 10
    theta = rng.uniform(-np.pi, np.pi, (n, b))
    h = rng.uniform(-1, 1, (n, b))
    dphi = rng.uniform(-np.pi, np.pi, (n, b))
    field = np.stack([np.cos(theta) * np.cos(h), np.sin(theta) * np.cos(h),
                      np.sin(h), np.cos(dphi), np.sin(dphi)], axis=-1)
    act = rng.uniform(0, 3, b)
    out = weather.local_intensity(field, act)
    assert out.shape == (n,)
    assert np.all(out >= 0.0) and np.all(np.isfinite(out))
