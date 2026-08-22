import numpy as np

from kalachakra.losses import reference as rl


def _random_field(shape, seed=0):
    """Build a valid local field tensor (..., 5) with unit sub-vectors."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(-np.pi, np.pi, size=shape)
    h = rng.uniform(-np.pi / 2, np.pi / 2, size=shape)
    dphi = rng.uniform(-np.pi, np.pi, size=shape)
    cos_h = np.cos(h)
    return np.stack(
        [np.cos(theta) * cos_h, np.sin(theta) * cos_h, np.sin(h),
         np.cos(dphi), np.sin(dphi)],
        axis=-1,
    )


def test_geodesic_loss_near_zero_for_identical_fields():
    # The clamped arccos (eps=1e-7) has a small floor ~sqrt(2*eps) ~= 4.5e-4 for
    # identical unit vectors; that is intended numerical-stability behavior.
    f = _random_field((4, 10))
    assert rl.geodesic_reconstruction_loss(f, f) < 1e-3


def test_geodesic_loss_increases_with_difference():
    a = _random_field((8, 10), seed=1)
    b = _random_field((8, 10), seed=2)
    small = 0.99 * a + 0.01 * b
    assert (rl.geodesic_reconstruction_loss(a, small)
            < rl.geodesic_reconstruction_loss(a, b))


def test_spectral_loss_zero_for_identical_sequences():
    seq = _random_field((16, 5), seed=3)
    assert rl.spectral_harmonic_loss(seq, seq) < 1e-6


def test_aspect_loss_is_rotation_invariant():
    rng = np.random.default_rng(4)
    lons = rng.uniform(-np.pi, np.pi, size=(10,))
    rotated = lons + 0.7  # rigid global rotation of the whole configuration
    assert rl.aspect_relational_invariance_loss(lons, rotated) < 1e-9


def test_aspect_loss_detects_structural_change():
    rng = np.random.default_rng(5)
    lons = rng.uniform(-np.pi, np.pi, size=(10,))
    perturbed = lons.copy()
    perturbed[0] += 1.0  # change one body's aspects to all others
    assert rl.aspect_relational_invariance_loss(lons, perturbed) > 1e-2


def test_composite_bundles_all_terms():
    a = _random_field((8, 10), seed=6)
    b = _random_field((8, 10), seed=7)
    lons_a = np.linspace(-np.pi, np.pi, 10, endpoint=False)
    lons_b = lons_a + np.linspace(0, 0.5, 10)
    parts = rl.composite_loss(a, b, recon_lons=lons_a, target_lons=lons_b)
    assert set(parts) == {"geodesic", "spectral", "aspect", "total"}
    assert abs(parts["total"]
               - (parts["geodesic"] + parts["spectral"] + parts["aspect"])) < 1e-9
