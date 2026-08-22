import numpy as np

from kalachakra import geometry as geo


def test_unit_vector_has_unit_norm():
    lon = np.array([0.0, 1.0, 2.0, -1.5])
    lat = np.array([0.0, 0.5, -0.3, 1.0])
    v = geo.to_unit_vector(lon, lat)
    norms = np.linalg.norm(v, axis=-1)
    assert np.allclose(norms, 1.0)


def test_geodesic_distance_identity_and_antipode():
    u = geo.to_unit_vector(np.array([0.3]), np.array([0.2]))
    assert geo.geodesic_distance(u, u)[0] < 1e-3
    antipode = -u
    assert abs(geo.geodesic_distance(u, antipode)[0] - np.pi) < 1e-2


def test_wrap_angle_range():
    theta = np.array([0.0, 3 * np.pi, -3 * np.pi, 7.0])
    w = geo.wrap_angle(theta)
    assert np.all(w >= -np.pi) and np.all(w < np.pi)


def test_angular_separation_symmetry():
    a, b = 0.1, 6.0  # b ~ 6.0 rad, close to 2pi
    assert np.isclose(geo.angular_separation(a, b), geo.angular_separation(b, a))
    assert geo.angular_separation(a, b) <= np.pi


def test_gmst_in_range():
    jd = np.array([2451545.0, 2459204.0, 1213073.5])
    g = geo.greenwich_mean_sidereal_time_deg(jd)
    assert np.all((g >= 0.0) & (g < 360.0))


def test_ecliptic_to_equatorial_zero_point():
    # At ecliptic (lon=0, lat=0) RA and Dec should both be ~0.
    eps = geo.obliquity_of_ecliptic(2451545.0)
    ra, dec = geo.ecliptic_to_equatorial(np.array([0.0]), np.array([0.0]), eps)
    assert abs(dec[0]) < 1e-6
    assert min(abs(ra[0]), abs(ra[0] - 2 * np.pi)) < 1e-6


def test_pairwise_matrix_is_symmetric_with_zero_diagonal():
    lons = np.array([0.0, 1.0, 2.5, 4.0])
    m = geo.pairwise_angular_matrix(lons)
    assert np.allclose(m, m.T)
    assert np.allclose(np.diag(m), 0.0)


def test_obliquity_is_about_23_4_degrees_at_j2000():
    eps = geo.obliquity_of_ecliptic(2451545.0)
    assert abs(np.rad2deg(eps) - 23.4393) < 1e-3
