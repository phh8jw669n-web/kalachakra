import numpy as np

import kalachakra.constants as C
from kalachakra.ephemeris.global_state import encode_body
from kalachakra.grid import geodesic
from kalachakra.projection import spatial


def _synthetic_global_frame(seed: int = 0) -> np.ndarray:
    """Build a plausible G(t) frame from random ecliptic states."""
    rng = np.random.default_rng(seed)
    rows = np.empty((C.N_BODIES, C.GLOBAL_BODY_FEATURES))
    for i in range(C.N_BODIES):
        lam = rng.uniform(-np.pi, np.pi)
        bet = rng.uniform(-0.1, 0.1)
        rows[i] = encode_body(lam, bet, r=rng.uniform(0.4, 30.0),
                              lam_dot=rng.normal(), bet_dot=rng.normal() * 0.01,
                              r_dot=rng.normal() * 0.01)
    return rows


def test_decode_ecliptic_roundtrip():
    lam = np.array([0.3, -1.2, 2.9])
    bet = np.array([0.05, -0.02, 0.1])
    frame = np.stack([encode_body(l, b, 1, 0, 0, 0) for l, b in zip(lam, bet)])
    dlon, dlat = spatial.decode_ecliptic(frame)
    assert np.allclose(dlon, lam, atol=1e-9)
    assert np.allclose(dlat, bet, atol=1e-9)


def test_projection_shape_and_encoding():
    grid = geodesic.fibonacci_sphere(200)
    frame = _synthetic_global_frame()
    field = spatial.project(frame, jd_ut=2451545.0, grid=grid)
    assert field.shape == (200, C.N_BODIES, C.LOCAL_BODY_FEATURES)

    # First three components form a unit vector (cos t cos h, sin t cos h, sin h).
    direction = field[..., :3]
    assert np.allclose(np.linalg.norm(direction, axis=-1), 1.0, atol=1e-6)

    # Last two components are (cos dphi, sin dphi) -> unit circle.
    circle = field[..., 3:5]
    assert np.allclose(np.linalg.norm(circle, axis=-1), 1.0, atol=1e-6)


def test_altitude_component_in_range():
    grid = geodesic.fibonacci_sphere(300)
    frame = _synthetic_global_frame(3)
    field = spatial.project(frame, jd_ut=2459204.0, grid=grid)
    sin_h = field[..., 2]
    assert np.all(sin_h >= -1.0 - 1e-9) and np.all(sin_h <= 1.0 + 1e-9)


def test_projection_is_deterministic():
    grid = geodesic.fibonacci_sphere(100)
    frame = _synthetic_global_frame(7)
    a = spatial.project(frame, 2451545.0, grid)
    b = spatial.project(frame, 2451545.0, grid)
    assert np.array_equal(a, b)


def test_topocentric_parallax_localizes_the_2024_eclipse():
    """The projection is topocentric: lunar parallax makes the Sun-Moon sky
    separation vary by ~1 deg across the globe (it would be constant if the field
    were geocentric), and the point of closest alignment falls on the real
    2024-04-08 totality track over northern Mexico."""
    import pytest

    from kalachakra.ephemeris import global_state
    from kalachakra.ephemeris.calendar import parse_datetime
    if not global_state.ephemeris_available():
        pytest.skip("pyswisseph not installed")

    jd = parse_datetime("2024-04-08T18:17:00Z")
    g = global_state.global_state_frame(jd)
    grid = geodesic.fibonacci_sphere(6000)
    field = spatial.project(g, jd, grid)

    # Sun (row 0) vs Moon (row 1) local sky directions -> angular separation.
    sun, moon = field[:, 0, :3], field[:, 1, :3]
    sep = np.degrees(np.arccos(np.clip(np.sum(sun * moon, axis=1), -1.0, 1.0)))

    # Geocentrically this separation is identical for every observer; only
    # topocentric parallax makes it vary. ~0.95 deg lunar parallax -> ~1 deg range.
    assert sep.max() - sep.min() > 0.5
    # Somewhere on Earth the disks align to near-zero (the totality track).
    assert sep.min() < 0.05
    # ...and that somewhere is over the Americas in daylight (real 2024 track).
    i = int(sep.argmin())
    assert 10.0 < np.degrees(grid.lat[i]) < 40.0
    assert -120.0 < np.degrees(grid.lon[i]) < -90.0


def test_parallax_masked_for_nodes_and_precession():
    """Parallax must apply to physical bodies only. The lunar nodes (Rahu/Ketu)
    and Ayanamsha are geometric directions at infinity — Swiss Ephemeris parks the
    Moon's distance in the node distance slot, so an unmasked topocentric
    correction would swing them ~1 deg across the globe. Guards that regression:
    the Sun-node sky separation is observer-independent (only the Sun's own tiny
    parallax leaks in) while the Sun-Moon separation varies by ~1 deg."""
    import pytest

    from kalachakra.ephemeris import bodies, global_state
    from kalachakra.ephemeris.calendar import parse_datetime
    if not global_state.ephemeris_available():
        pytest.skip("pyswisseph not installed")

    jd = parse_datetime("2024-04-08T18:17:00Z")
    g = global_state.global_state_frame(jd)
    grid = geodesic.fibonacci_sphere(4000)
    field = spatial.project(g, jd, grid)

    def spread(a, b):
        d = np.sum(field[:, a, :3] * field[:, b, :3], axis=1)
        s = np.degrees(np.arccos(np.clip(d, -1.0, 1.0)))
        return float(s.max() - s.min())

    sun = bodies.index_of("Sun")
    # Moon is physical -> real parallax makes its separation from the Sun vary.
    assert spread(sun, bodies.index_of("Moon")) > 0.3
    # Nodes / precession carry no parallax -> observer-independent (< the Sun's own).
    for name in ("Rahu", "Ketu", "Ayanamsha"):
        assert spread(sun, bodies.index_of(name)) < 0.05
