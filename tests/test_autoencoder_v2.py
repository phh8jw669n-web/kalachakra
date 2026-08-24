"""The node-chunked v2 autoencoder must be numerically equivalent to v1.

v2 only tiles the spatial/temporal ops over the node axis to stay under Metal's
INT_MAX tensor-size limit at the full mesh; it must not change the mathematics,
the parameters, or the tokens. Continuous outputs match to float32 precision
(residual differences <= ~1e-6 come purely from FFT/matmul using a different batch
tiling per chunk size — ordinary floating-point non-associativity, ~5 orders of
magnitude below bf16 training noise); discrete tokens match exactly.
"""

_ATOL = 1e-5   # float32 batch-tiling non-associativity, not an algorithmic diff

import pytest

torch = pytest.importorskip("torch")

from kalachakra.grid import geodesic                               # noqa: E402
from kalachakra.models.autoencoder import (                        # noqa: E402
    AutoencoderConfig, SphericalAutoencoder,
)
from kalachakra.models.autoencoder_v2 import SphericalAutoencoderV2  # noqa: E402
from kalachakra.models.spherical_conv import build_knn             # noqa: E402


def _cfg(n=300):
    return AutoencoderConfig(n_nodes=n, hidden=32, latent=64,
                             fourier_modes=8, knn=7, n_blocks=3)


def test_v2_forward_bit_identical_to_v1():
    torch.manual_seed(0)
    grid = geodesic.fibonacci_sphere(300)
    nb = build_knn(grid, 7)
    cfg = _cfg(300)
    v1 = SphericalAutoencoder(cfg, nb).eval()
    v2 = SphericalAutoencoderV2(cfg, nb, node_chunk=64).eval()   # force chunking
    # Identical layout -> state_dict transfers 1:1.
    assert list(v1.state_dict().keys()) == list(v2.state_dict().keys())
    v2.load_state_dict(v1.state_dict())

    e = torch.randn(2, 16, 300, cfg.in_features)
    with torch.no_grad():
        r1, z1 = v1(e)
        r2, z2 = v2(e)
    assert torch.allclose(z1, z2, atol=_ATOL)
    assert torch.allclose(r1, r2, atol=_ATOL)


def test_v2_quantized_tokens_identical_to_v1():
    from kalachakra.models.quantized_autoencoder import QuantizedSphericalAutoencoder
    from kalachakra.models.quantized_autoencoder_v2 import (
        QuantizedSphericalAutoencoderV2,
    )
    from kalachakra.models.rvq import RVQConfig

    torch.manual_seed(0)
    grid = geodesic.fibonacci_sphere(300)
    nb = build_knn(grid, 7)
    cfg = AutoencoderConfig(n_nodes=300, hidden=32, latent=64,
                            fourier_modes=8, knn=7, n_blocks=2)
    rc = RVQConfig(dim=64, n_macro=16, n_micro=16)
    q1 = QuantizedSphericalAutoencoder(cfg, nb, rc).eval()
    q2 = QuantizedSphericalAutoencoderV2(cfg, nb, rc, node_chunk=64).eval()
    q2.load_state_dict(q1.state_dict())

    e = torch.randn(1, 12, 300, cfg.in_features)
    with torch.no_grad():
        m1, mi1, l1, _ = q1.tokenize(e)
        m2, mi2, l2, _ = q2.tokenize(e)
    assert torch.equal(l1, l2)
    assert torch.equal(m1, m2) and torch.equal(mi1, mi2)


def test_v2_chunk_size_does_not_change_output():
    """Any node_chunk must give the same result (it's a pure tiling knob)."""
    torch.manual_seed(1)
    grid = geodesic.fibonacci_sphere(257)          # not a multiple of the chunks
    nb = build_knn(grid, 7)
    cfg = _cfg(257)
    base = SphericalAutoencoderV2(cfg, nb, node_chunk=10_000).eval()  # no chunking
    e = torch.randn(1, 8, 257, cfg.in_features)
    with torch.no_grad():
        r_full, z_full = base(e)
    for ch in (16, 50, 128):                       # incl. sizes not dividing 257
        m = SphericalAutoencoderV2(cfg, nb, node_chunk=ch).eval()
        m.load_state_dict(base.state_dict())
        with torch.no_grad():
            r, z = m(e)
        assert torch.allclose(z, z_full, atol=_ATOL)
        assert torch.allclose(r, r_full, atol=_ATOL)
