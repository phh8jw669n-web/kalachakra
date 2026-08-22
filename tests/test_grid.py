import numpy as np

import kalachakra.constants as C
from kalachakra.grid import geodesic


def test_fibonacci_exact_count_and_on_sphere():
    g = geodesic.fibonacci_sphere(5000)
    assert g.n_nodes == 5000
    assert np.allclose(np.linalg.norm(g.xyz, axis=1), 1.0, atol=1e-9)


def test_default_grid_hits_blueprint_node_count():
    g = geodesic.default_grid()
    assert g.n_nodes == C.N_SPATIAL_NODES == 122_880


def test_latlon_ranges():
    g = geodesic.fibonacci_sphere(2000)
    assert np.all(g.lat >= -np.pi / 2 - 1e-9) and np.all(g.lat <= np.pi / 2 + 1e-9)
    assert np.all(g.lon >= -np.pi - 1e-9) and np.all(g.lon <= np.pi + 1e-9)


def test_icosphere_vertex_count_formula():
    for n in range(4):
        g = geodesic.icosphere(n)
        assert g.n_nodes == geodesic.icosphere_vertex_count(n)
        assert np.allclose(np.linalg.norm(g.xyz, axis=1), 1.0, atol=1e-9)


def test_fibonacci_is_reasonably_uniform():
    # Mean z should be ~0 and coverage should span both hemispheres.
    g = geodesic.fibonacci_sphere(10000)
    assert abs(g.xyz[:, 2].mean()) < 1e-2
    assert g.xyz[:, 2].min() < -0.99 and g.xyz[:, 2].max() > 0.99
