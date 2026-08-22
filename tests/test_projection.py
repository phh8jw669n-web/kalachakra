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
