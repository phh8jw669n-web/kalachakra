"""Model checkpoint save/load roundtrip. Skipped when torch is absent."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from kalachakra.grid import geodesic                               # noqa: E402
from kalachakra.models.autoencoder import (                        # noqa: E402
    AutoencoderConfig, SphericalAutoencoder,
)
from kalachakra.models.spherical_conv import build_knn             # noqa: E402
from kalachakra.training.checkpoint import load_model, save_model  # noqa: E402


def test_save_load_roundtrip_preserves_outputs(tmp_path):
    grid = geodesic.fibonacci_sphere(48)
    neighbors = build_knn(grid, 6)
    cfg = AutoencoderConfig(n_nodes=48, hidden=16, latent=64,
                            fourier_modes=4, knn=6, n_blocks=1)
    model = SphericalAutoencoder(cfg, neighbors).eval()

    e = torch.randn(1, 8, 48, cfg.in_features)
    with torch.no_grad():
        z_before = model.encode(e)

    path = save_model(tmp_path / "m.pt", model, cfg, neighbors, grid_xyz=grid.xyz)
    assert path.exists()

    reloaded, cfg2, grid_xyz = load_model(path)
    assert cfg2.latent == 64 and cfg2.n_nodes == 48
    assert grid_xyz.shape == (48, 3)
    with torch.no_grad():
        z_after = reloaded.encode(e)
    assert torch.allclose(z_before, z_after, atol=1e-6)
