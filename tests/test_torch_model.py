"""Smoke tests for the PyTorch neural core. Skipped when torch is absent."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import kalachakra.constants as C                                    # noqa: E402
from kalachakra.grid import geodesic                               # noqa: E402
from kalachakra.losses.geometric import CompositeGeodesicLoss      # noqa: E402
from kalachakra.models.autoencoder import (                        # noqa: E402
    AutoencoderConfig, SphericalAutoencoder,
)
from kalachakra.models.fno import SpectralConv1d                   # noqa: E402
from kalachakra.models.spherical_conv import GeodesicConv, build_knn  # noqa: E402
from kalachakra.training.optim import Lion                         # noqa: E402


def _tiny_model(n_nodes=64, knn=6):
    grid = geodesic.fibonacci_sphere(n_nodes)
    neighbors = build_knn(grid, knn)
    cfg = AutoencoderConfig(n_nodes=n_nodes, hidden=32, latent=C.LATENT_DIM,
                            fourier_modes=4, knn=knn, n_blocks=2)
    return SphericalAutoencoder(cfg, neighbors), grid


def test_spectral_conv_preserves_length():
    layer = SpectralConv1d(3, 5, modes=4)
    x = torch.randn(2, 3, 16)
    y = layer(x)
    assert y.shape == (2, 5, 16)


def test_geodesic_conv_shapes():
    grid = geodesic.fibonacci_sphere(50)
    neighbors = build_knn(grid, 5)
    conv = GeodesicConv(8, 12, neighbors)
    x = torch.randn(3, 50, 8)
    assert conv(x).shape == (3, 50, 12)


def test_build_knn_includes_self_and_shape():
    grid = geodesic.fibonacci_sphere(40)
    idx = build_knn(grid, 6)
    assert idx.shape == (40, 6)
    # Each node's own index should appear among its nearest neighbors.
    assert all(i in idx[i] for i in range(40))


def test_autoencoder_roundtrip_shapes():
    model, _ = _tiny_model(n_nodes=64)
    b, t, n, f = 2, 8, 64, C.LOCAL_FIELD_WIDTH
    e = torch.randn(b, t, n, f)
    recon, z = model(e)
    assert recon.shape == (b, t, n, f)
    assert z.shape == (b, t, n, C.LATENT_DIM)


def test_backward_and_lion_step_change_weights():
    model, _ = _tiny_model(n_nodes=64)
    opt = Lion(model.parameters(), lr=1e-3, weight_decay=0.01)
    e = torch.randn(2, 8, 64, C.LOCAL_FIELD_WIDTH)
    crit = CompositeGeodesicLoss()

    before = [p.detach().clone() for p in model.parameters()]
    recon, _z = model(e)
    total, parts = crit(recon.unflatten(-1, (-1, 5)), e.unflatten(-1, (-1, 5)))
    assert torch.isfinite(total)
    total.backward()
    opt.step()

    after = list(model.parameters())
    changed = any(not torch.equal(b, a) for b, a in zip(before, after))
    assert changed, "Lion step should update parameters"
    assert set(parts) >= {"geodesic", "spectral", "total"}


def test_spectral_loss_accepts_bfloat16():
    # Regression: under autocast the loss receives bf16 tensors; rfft must not be
    # called on bf16 (errors on MPS). The loss casts to float32 internally.
    from kalachakra.losses.geometric import spectral_harmonic_loss
    seq = torch.randn(1, 16, 8, 5, dtype=torch.bfloat16)
    out = spectral_harmonic_loss(seq, seq)
    assert torch.isfinite(out) and out.dtype == torch.float32


def test_composite_loss_accepts_bfloat16():
    crit = CompositeGeodesicLoss()
    field = torch.randn(1, 16, 8, 10, 5, dtype=torch.bfloat16)
    total, parts = crit(field, field)
    assert torch.isfinite(total)


def test_composite_loss_zero_floor_for_identical():
    crit = CompositeGeodesicLoss()
    # Build a valid field with unit sub-vectors.
    rng = np.random.default_rng(0)
    theta = torch.tensor(rng.uniform(-np.pi, np.pi, size=(1, 4, 8, 10)))
    h = torch.tensor(rng.uniform(-1, 1, size=(1, 4, 8, 10)))
    dphi = torch.tensor(rng.uniform(-np.pi, np.pi, size=(1, 4, 8, 10)))
    field = torch.stack(
        [torch.cos(theta) * torch.cos(h), torch.sin(theta) * torch.cos(h),
         torch.sin(h), torch.cos(dphi), torch.sin(dphi)], dim=-1,
    ).float()
    total, _ = crit(field, field)
    assert float(total) < 1e-2
