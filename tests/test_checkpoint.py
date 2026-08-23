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


def test_checkpoint_stamps_projection_version_and_warns_on_mismatch(tmp_path):
    import warnings

    from kalachakra import constants as C
    from kalachakra.training.checkpoint import load_model

    grid = geodesic.fibonacci_sphere(48)
    neighbors = build_knn(grid, 6)
    cfg = AutoencoderConfig(n_nodes=48, hidden=16, latent=64,
                            fourier_modes=4, knn=6, n_blocks=1)
    model = SphericalAutoencoder(cfg, neighbors).eval()
    path = save_model(tmp_path / "m.pt", model, cfg, neighbors, grid_xyz=grid.xyz)

    # Stamped with the current projection version; matching load is silent.
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    assert ckpt["projection_version"] == C.PROJECTION_VERSION
    with warnings.catch_warnings():
        warnings.simplefilter("error")            # no warning on a matching load
        load_model(path)

    # A checkpoint from a different projection warns loudly on load.
    ckpt["projection_version"] = C.PROJECTION_VERSION + 99
    torch.save(ckpt, path)
    with pytest.warns(RuntimeWarning, match="projection"):
        load_model(path)


def test_quantized_model_roundtrip(tmp_path):
    from kalachakra.models.autoencoder import AutoencoderConfig
    from kalachakra.models.quantized_autoencoder import QuantizedSphericalAutoencoder
    from kalachakra.models.rvq import RVQConfig
    from kalachakra.training.checkpoint import load_quantized_model, save_quantized_model

    grid = geodesic.fibonacci_sphere(48)
    neighbors = build_knn(grid, 6)
    ae_cfg = AutoencoderConfig(n_nodes=48, hidden=16, latent=64,
                               fourier_modes=4, knn=6, n_blocks=1)
    rvq_cfg = RVQConfig(dim=64, n_macro=16, n_micro=16)
    model = QuantizedSphericalAutoencoder(ae_cfg, neighbors, rvq_cfg).eval()

    e = torch.randn(1, 8, 48, ae_cfg.in_features)
    with torch.no_grad():
        _m0, _mi0, leaf0, _q0 = model.tokenize(e)

    path = save_quantized_model(tmp_path / "q.pt", model, ae_cfg, neighbors,
                                rvq_cfg, grid_xyz=grid.xyz)
    reloaded, ae2, rvq2, xyz = load_quantized_model(path)
    assert rvq2.n_leaf == 256 and xyz.shape == (48, 3)
    with torch.no_grad():
        _m1, _mi1, leaf1, _q1 = reloaded.tokenize(e)
    # deterministic tokenization survives the round trip
    assert torch.equal(leaf0, leaf1)
