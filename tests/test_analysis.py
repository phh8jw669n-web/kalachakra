import numpy as np

from kalachakra.analysis import anomaly, signatures
from kalachakra.analysis.clustering import cluster_latents
from kalachakra.grid import geodesic
from kalachakra.serving.broadcast import BroadcastEngine


def test_geometric_potential_is_l2_norm():
    z = np.array([[3.0, 4.0], [0.0, 0.0], [1.0, 0.0]])
    p = signatures.geometric_potential_field(z)
    assert np.allclose(p, [5.0, 0.0, 1.0])


def test_temporal_shear_flags_a_jump():
    # A latent field that is constant then jumps -> shear peaks at the jump.
    T, N, D = 10, 1, 4
    z = np.zeros((T, N, D))
    z[5:] = 1.0
    shear = signatures.temporal_shear_gradient(z, time_axis=0)
    assert shear.shape == (T, N)
    assert shear[:, 0].argmax() in (4, 5)


def test_singularity_detection_finds_injected_spike():
    rng = np.random.default_rng(0)
    potential = rng.normal(size=(20, 30)) * 0.1 + 1.0
    shear = rng.normal(size=(20, 30)) * 0.1 + 1.0
    potential[7, 12] = 20.0
    shear[7, 12] = 20.0
    events = anomaly.detect_singularities(potential, shear, sigma=4.0)
    assert events, "expected at least one singularity"
    assert (events[0].time_index, events[0].node_index) == (7, 12)


def test_clustering_returns_labels():
    rng = np.random.default_rng(1)
    blob_a = rng.normal(loc=0.0, scale=0.1, size=(120, 8))
    blob_b = rng.normal(loc=5.0, scale=0.1, size=(120, 8))
    z = np.vstack([blob_a, blob_b])
    result = cluster_latents(z, min_cluster_size=20)
    assert result.labels.shape[0] == z.shape[0]
    assert result.n_clusters >= 1


def test_broadcast_engine_query():
    grid = geodesic.fibonacci_sphere(500)
    potential = np.linspace(0, 1, 500)
    shear = np.linspace(1, 0, 500)
    engine = BroadcastEngine(grid, potential, shear)
    reading = engine.query(lat_deg=45.0, lon_deg=10.0)
    assert 0 <= reading.node_index < 500
    hm = engine.heatmap()
    assert hm["potential"].shape == (500,)
