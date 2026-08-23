import numpy as np

from kalachakra.projection.microgrid import bbox_microgrid, resolution_km


def test_microgrid_shape_and_on_sphere():
    g = bbox_microgrid(20, 70, 30, 80, density=16)
    assert g.n_nodes == 16 * 16
    assert np.allclose(np.linalg.norm(g.xyz, axis=1), 1.0, atol=1e-9)
    assert np.rad2deg(g.lat).min() >= 20 - 1e-6
    assert np.rad2deg(g.lat).max() <= 30 + 1e-6


def test_microgrid_density_increases_resolution():
    coarse = resolution_km(20, 70, 30, 80, 16)
    fine = resolution_km(20, 70, 30, 80, 128)
    assert fine < coarse and fine > 0


def test_microgrid_rejects_bad_bbox():
    import pytest
    with pytest.raises(ValueError):
        bbox_microgrid(30, 70, 20, 80, density=8)   # min_lat > max_lat
