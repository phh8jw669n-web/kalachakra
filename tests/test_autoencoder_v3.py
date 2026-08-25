"""Discrete VQ-bottleneck autoencoder (v3): stable quantizer, STE, EMA, restart.

The v3 model is a VQ-VAE with the modern stability formulation: L2-normalized
(cosine) lookup, an EMA-updated codebook (a buffer, not gradient-trained),
commitment-only loss, and dead-code restart. These tests pin that contract.
"""

import warnings

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F                                       # noqa: E402

from kalachakra.grid import geodesic                                  # noqa: E402
from kalachakra.models.autoencoder_v3 import (                        # noqa: E402
    SpectralConv1dV3, VectorQuantizer, VQAutoencoderV3,
    VQAutoencoderV3Config, build_knn,
)


def _model(n=300, blocks=2, node_chunk=64, grad_checkpoint=False, codebook=4096):
    grid = geodesic.fibonacci_sphere(n)
    nb = build_knn(grid.xyz, 7)
    cfg = VQAutoencoderV3Config(
        n_nodes=n, in_features=50, hidden=32, latent=64, fourier_modes=8, knn=7,
        n_blocks=blocks, codebook_size=codebook, commitment_beta=0.25,
        node_chunk=node_chunk, vq_chunk=4096, grad_checkpoint=grad_checkpoint,
    )
    return VQAutoencoderV3(cfg, nb), grid


def _spectral_ref(conv, x):
    """The original rfft/irfft spectral path, for the length-1 equivalence check."""
    b, _, length = x.shape
    xf = x.float()
    x_ft = torch.fft.rfft(xf, dim=-1)
    keep = min(conv.modes, x_ft.shape[-1])
    xr, xi = x_ft.real[:, :, :keep], x_ft.imag[:, :, :keep]
    wr = conv.weight[:, :, :keep, 0].float()
    wi = conv.weight[:, :, :keep, 1].float()
    out_r = torch.einsum("bix,iox->box", xr, wr) - torch.einsum("bix,iox->box", xi, wi)
    out_i = torch.einsum("bix,iox->box", xr, wi) + torch.einsum("bix,iox->box", xi, wr)
    out_ft = torch.zeros(b, conv.out_channels, x_ft.shape[-1], dtype=torch.cfloat)
    out_ft[:, :, :keep] = torch.complex(out_r, out_i)
    return torch.fft.irfft(out_ft, n=length, dim=-1)


def test_spectral_length1_matches_fft_and_is_warning_free():
    # single-frame inference (length==1) must be bit-identical to the rfft/irfft
    # round trip, and must not emit the benign MPS out-resize UserWarning.
    torch.manual_seed(0)
    conv = SpectralConv1dV3(8, 5, modes=6).eval()
    x = torch.randn(2048, 8, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)     # any UserWarning -> failure
        with torch.no_grad():
            y = conv(x)
    with torch.no_grad():
        ref = _spectral_ref(conv, x)
    assert y.shape == (2048, 5, 1)
    assert torch.equal(y, ref)                          # bit-identical


def test_model_single_frame_encode_is_warning_free():
    model, _ = _model(n=200, blocks=2)
    model.eval()
    e = torch.randn(1, 1, 200, 50)                      # (B, T=1, N, 50) single frame
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        with torch.no_grad():
            z = model.encode(e)
    assert z.shape == (1, 1, 200, 64)


def test_forward_signature_and_shapes():
    m, _ = _model()
    e = torch.randn(2, 12, 300, 50)
    recon, z, idx, vq_loss = m(e)          # exactly the 4-tuple the spec requires
    assert recon.shape == e.shape
    assert z.shape == (2, 12, 300, 64)
    assert idx.shape == (2, 12, 300)
    assert idx.dtype == torch.long and 0 <= int(idx.min()) and int(idx.max()) < 4096
    assert vq_loss.ndim == 0 and torch.isfinite(vq_loss)


def test_straight_through_and_ema_not_gradient():
    m, _ = _model()
    m.train()
    cb_before = m.vq.codebook.clone()
    e = torch.randn(1, 8, 300, 50)
    recon, _z, _idx, vq_loss = m(e)
    (recon.pow(2).mean() + vq_loss).backward()
    # STE: reconstruction gradient reaches the encoder despite the argmax lookup.
    assert m.to_latent.weight.grad is not None and m.to_latent.weight.grad.abs().sum() > 0
    assert m.project.weight.grad.abs().sum() > 0          # decoder trained by recon
    # Codebook is a buffer updated by EMA, NOT by gradient.
    assert m.vq.codebook.grad is None
    assert not torch.equal(cb_before, m.vq.codebook)      # EMA moved it in forward


def test_cosine_quantization_unit_norm_and_commitment_loss():
    torch.manual_seed(0)
    vq = VectorQuantizer(dim=64, codebook_size=256, beta=0.25, vq_chunk=1024).eval()
    z_e = torch.randn(500, 64)
    z_q, idx, vq_loss, _ppl = vq(z_e)
    # everything lives on the unit sphere.
    assert torch.allclose(z_q.norm(dim=1), torch.ones(500), atol=1e-5)
    # loss is beta * commitment in the NORMALIZED space, no codebook term.
    z_norm = F.normalize(z_e, dim=1)
    e_norm = F.normalize(vq.codebook, dim=1)
    zq_ref = F.embedding(idx, e_norm)
    assert torch.allclose(z_q, zq_ref, atol=1e-6)         # forward is the unit code
    assert torch.allclose(vq_loss, 0.25 * F.mse_loss(z_norm, zq_ref), atol=1e-6)


def test_perplexity_range_and_collapse():
    vq = VectorQuantizer(dim=8, codebook_size=16, vq_chunk=64).eval()
    # identical directions -> one code -> perplexity ~1 (collapse).
    same = torch.ones(100, 8)
    _zq, idx, _l, ppl = vq(same)
    assert idx.unique().numel() == 1 and 0.99 <= float(ppl) <= 1.01
    # diverse directions -> several codes -> perplexity > 1, bounded by K.
    _zq, _i, _l, ppl2 = vq(torch.randn(500, 8))
    assert 1.0 < float(ppl2) <= vq.codebook_size


def test_dead_code_restart_reseeds_unused_codes():
    torch.manual_seed(0)
    vq = VectorQuantizer(dim=4, codebook_size=64, restart_after=1, vq_chunk=64).train()
    # A handful of near-identical inputs uses only a few codes; the rest are unused.
    z = torch.randn(20, 4)
    cb_before = vq.codebook.clone()
    _zq, idx, _l, _p = vq(z)
    # With restart_after=1, every code unused this step is reseeded immediately, so
    # all usage counters reset and many codebook rows change.
    assert float(vq.unused_steps.max()) == 0.0
    assert not torch.equal(cb_before, vq.codebook)
    assert torch.allclose(vq.codebook.norm(dim=1), torch.ones(64), atol=1e-5)


def test_codebook_and_ema_saved_in_state_dict_and_reloads():
    m, _ = _model()
    sd = m.state_dict()
    for k in ("vq.codebook", "vq.cluster_size", "vq.embed_avg", "vq.unused_steps"):
        assert k in sd
    assert tuple(sd["vq.codebook"].shape) == (4096, 64)
    m2, _ = _model()
    m2.load_state_dict(sd)
    e = torch.randn(1, 6, 300, 50)
    with torch.no_grad():
        assert torch.equal(m.tokenize(e), m2.tokenize(e))   # tokenize is eval (no EMA)


def test_grad_checkpoint_is_exact():
    torch.manual_seed(0)
    m0, _ = _model(blocks=3, grad_checkpoint=False)
    m1, _ = _model(blocks=3, grad_checkpoint=True)
    m1.load_state_dict(m0.state_dict())
    m0.train()
    m1.train()
    e = torch.randn(1, 8, 300, 50)
    r0, _z0, _i0, v0 = m0(e)
    (r0.pow(2).mean() + v0).backward()
    r1, _z1, _i1, v1 = m1(e)
    (r1.pow(2).mean() + v1).backward()
    assert torch.equal(r0, r1)
    for p0, p1 in zip(m0.parameters(), m1.parameters()):
        assert torch.equal(p0.grad, p1.grad)


def test_node_chunk_is_result_invariant():
    torch.manual_seed(1)
    m_full, _ = _model(n=257, node_chunk=10_000)
    m_full.eval()                                    # eval -> no EMA state changes
    e = torch.randn(1, 6, 257, 50)
    with torch.no_grad():
        r_full, _z, i_full, _ = m_full(e)
    for ch in (16, 64):
        m, _ = _model(n=257, node_chunk=ch)
        m.load_state_dict(m_full.state_dict())
        m.eval()
        with torch.no_grad():
            r, _z2, i, _ = m(e)
        assert torch.allclose(r, r_full, atol=1e-5)
        assert torch.equal(i, i_full)                # same tokens
