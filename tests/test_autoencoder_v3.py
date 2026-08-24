"""Discrete VQ-bottleneck autoencoder (v3): quantizer, STE, perplexity, saving.

The v3 model is a standalone VQ-VAE: a discrete 4096-entry codebook sits between
the encoder's ``to_latent`` and the decoder's ``from_latent``. These tests pin the
contract the request specifies — the 4-tuple forward signature, straight-through
gradients to BOTH the encoder and the codebook, the VQ loss terms, batch
perplexity, and codebook persistence in the state_dict.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from kalachakra.grid import geodesic                                  # noqa: E402
from kalachakra.models.autoencoder_v3 import (                        # noqa: E402
    VectorQuantizer, VQAutoencoderV3, VQAutoencoderV3Config, build_knn,
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


def test_forward_signature_and_shapes():
    m, _ = _model()
    e = torch.randn(2, 12, 300, 50)
    recon, z, idx, vq_loss = m(e)          # exactly the 4-tuple the spec requires
    assert recon.shape == e.shape
    assert z.shape == (2, 12, 300, 64)
    assert idx.shape == (2, 12, 300)
    assert idx.dtype == torch.long and 0 <= int(idx.min()) and int(idx.max()) < 4096
    assert vq_loss.ndim == 0 and torch.isfinite(vq_loss)


def test_straight_through_reaches_encoder_and_codebook():
    m, _ = _model()
    m.train()
    e = torch.randn(1, 8, 300, 50)
    recon, _z, _idx, vq_loss = m(e)
    (recon.pow(2).mean() + vq_loss).backward()
    # STE: reconstruction gradient must reach the encoder despite the argmin.
    assert m.to_latent.weight.grad is not None
    assert m.to_latent.weight.grad.abs().sum() > 0
    # codebook must be trained by the codebook-loss term.
    assert m.vq.codebook.grad is not None and m.vq.codebook.grad.abs().sum() > 0
    # decoder trained by the reconstruction.
    assert m.project.weight.grad.abs().sum() > 0


def test_vq_loss_terms_match_the_formula():
    torch.manual_seed(0)
    vq = VectorQuantizer(dim=64, codebook_size=256, beta=0.25, vq_chunk=1024)
    z_e = torch.randn(500, 64, requires_grad=True)
    z_q, idx, vq_loss, ppl = vq(z_e)
    # recompute the reference terms independently
    zq_ref = torch.nn.functional.embedding(idx, vq.codebook)
    codebook = torch.mean((zq_ref - z_e.detach()) ** 2)
    commit = torch.mean((z_e - zq_ref.detach()) ** 2)
    assert torch.allclose(vq_loss, codebook + 0.25 * commit, atol=1e-6)
    # forward value is the quantized vector (STE only changes the backward)
    assert torch.allclose(z_q, zq_ref, atol=1e-6)


def test_perplexity_in_valid_range_and_reflects_usage():
    # One code used -> perplexity 1; uniform over K -> perplexity K.
    vq = VectorQuantizer(dim=4, codebook_size=8, beta=0.25, vq_chunk=64)
    with torch.no_grad():
        vq.codebook.zero_()
        vq.codebook[0] = 10.0                     # make code 0 dominate for aligned inputs
    z_e = torch.zeros(100, 4)                      # all map to the nearest single code
    _zq, idx, _l, ppl = vq(z_e)
    assert idx.unique().numel() == 1
    assert 0.99 <= float(ppl) <= 1.01             # collapsed -> perplexity ~1
    assert 1.0 <= float(ppl) <= vq.codebook_size


def test_codebook_saved_in_state_dict_and_reloads():
    m, _ = _model()
    sd = m.state_dict()
    assert "vq.codebook" in sd and tuple(sd["vq.codebook"].shape) == (4096, 64)
    m2, _ = _model()
    m2.load_state_dict(sd)                          # exact reload
    e = torch.randn(1, 6, 300, 50)
    with torch.no_grad():
        assert torch.equal(m.tokenize(e), m2.tokenize(e))


def test_grad_checkpoint_is_exact():
    torch.manual_seed(0)
    m0, _ = _model(blocks=3, grad_checkpoint=False)
    m1, _ = _model(blocks=3, grad_checkpoint=True)
    m1.load_state_dict(m0.state_dict())
    m0.train(); m1.train()
    e = torch.randn(1, 8, 300, 50)
    r0, _z0, _i0, v0 = m0(e); (r0.pow(2).mean() + v0).backward()
    r1, _z1, _i1, v1 = m1(e); (r1.pow(2).mean() + v1).backward()
    assert torch.equal(r0, r1)
    for p0, p1 in zip(m0.parameters(), m1.parameters()):
        assert torch.equal(p0.grad, p1.grad)


def test_node_chunk_is_result_invariant():
    torch.manual_seed(1)
    m_full, _ = _model(n=257, node_chunk=10_000)    # no chunking
    e = torch.randn(1, 6, 257, 50)
    with torch.no_grad():
        r_full, z_full, i_full, _ = m_full(e)
    for ch in (16, 64):
        m, _ = _model(n=257, node_chunk=ch)
        m.load_state_dict(m_full.state_dict())
        with torch.no_grad():
            r, z, i, _ = m(e)
        assert torch.allclose(r, r_full, atol=1e-5)
        assert torch.equal(i, i_full)               # same tokens
