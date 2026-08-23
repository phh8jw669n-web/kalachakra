"""Smoke tests for the quantized autoencoder. Skipped when torch is absent."""

import pytest

torch = pytest.importorskip("torch")

import kalachakra.constants as C                                    # noqa: E402
from kalachakra.grid import geodesic                               # noqa: E402
from kalachakra.models.autoencoder import AutoencoderConfig        # noqa: E402
from kalachakra.models.quantized_autoencoder import (              # noqa: E402
    QuantizedSphericalAutoencoder,
)
from kalachakra.models.rvq import RVQConfig                        # noqa: E402
from kalachakra.models.spherical_conv import build_knn            # noqa: E402


def _model(n=64):
    grid = geodesic.fibonacci_sphere(n)
    neighbors = build_knn(grid, 6)
    ae_cfg = AutoencoderConfig(n_nodes=n, hidden=32, latent=64,
                               fourier_modes=4, knn=6, n_blocks=1)
    return QuantizedSphericalAutoencoder(ae_cfg, neighbors,
                                         RVQConfig(dim=64, n_macro=16, n_micro=16))


def test_forward_shapes_and_tokens():
    model = _model(64).eval()
    e = torch.randn(2, 8, 64, C.LOCAL_FIELD_WIDTH)
    recon, z, quantized, info = model(e)
    assert recon.shape == e.shape
    assert z.shape == (2, 8, 64, 64) and quantized.shape == z.shape
    assert info["leaf_idx"].shape == (2, 8, 64)
    assert info["leaf_idx"].max() < 16 * 16


def test_training_step_backward():
    model = _model(64).train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    e = torch.randn(2, 8, 64, C.LOCAL_FIELD_WIDTH)
    recon, z, quantized, info = model(e)
    recon_loss = torch.nn.functional.mse_loss(recon, e)
    total = recon_loss + info["vq_loss"]
    assert torch.isfinite(total)
    total.backward()
    # encoder receives gradient through the straight-through quantizer
    assert model.ae.lift.weight.grad is not None
    opt.step()


def test_tokenize_inference():
    model = _model(48)
    e = torch.randn(1, 8, 48, C.LOCAL_FIELD_WIDTH)
    macro, micro, leaf, quant = model.tokenize(e)
    assert macro.shape == (1, 8, 48)
    assert torch.equal(leaf, macro * 16 + micro)
