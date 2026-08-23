"""Tests for Hierarchical Residual VQ. Skipped when torch is absent."""

import pytest

torch = pytest.importorskip("torch")

from kalachakra.models.rvq import HierarchicalResidualVQ, RVQConfig  # noqa: E402


def test_leaf_count_is_4096():
    assert RVQConfig().n_leaf == 4096


def test_forward_shapes_and_index_ranges():
    rvq = HierarchicalResidualVQ(RVQConfig(dim=64)).eval()
    z = torch.randn(2, 5, 7, 64)
    q, info = rvq(z)
    assert q.shape == z.shape
    assert info["macro_idx"].shape == (2, 5, 7)
    assert info["leaf_idx"].max() < 4096 and info["leaf_idx"].min() >= 0
    # leaf = macro * n_micro + micro
    assert torch.equal(info["leaf_idx"],
                       info["macro_idx"] * 64 + info["micro_idx"])


def test_straight_through_gradient_reaches_encoder():
    rvq = HierarchicalResidualVQ(RVQConfig(dim=16, n_macro=8, n_micro=8))
    z = torch.randn(4, 16, requires_grad=True)
    q, info = rvq(z)
    (q.sum() + info["vq_loss"]).backward()
    assert z.grad is not None
    # straight-through: d quantized / d z == identity, so grad from q.sum() is ~1.
    assert torch.isfinite(z.grad).all()


def test_lookup_matches_forward():
    rvq = HierarchicalResidualVQ(RVQConfig(dim=32, n_macro=16, n_micro=16)).eval()
    z = torch.randn(10, 32)
    q, info = rvq(z)
    recon = rvq.lookup(info["leaf_idx"])
    assert torch.allclose(q, recon, atol=1e-5)


def test_ema_training_reduces_quantization_error():
    torch.manual_seed(0)
    cfg = RVQConfig(dim=8, n_macro=16, n_micro=16, decay=0.9)
    rvq = HierarchicalResidualVQ(cfg).train()
    # Data on a few tight clusters -> quantizer should learn to represent them.
    centers = torch.randn(6, 8) * 5
    def batch():
        idx = torch.randint(0, 6, (256,))
        return centers[idx] + 0.05 * torch.randn(256, 8)

    with torch.no_grad():
        q0, _ = rvq(batch())
        err0 = (q0 - batch()).pow(2).mean().item()  # rough initial scale
    for _ in range(50):
        rvq(batch())
    with torch.no_grad():
        data = batch()
        q1, _ = rvq(data)
        err1 = (q1 - data).pow(2).mean().item()
    assert err1 < err0  # codebook adapted to the data


def test_rvq_accepts_bfloat16_in_training():
    # Regression: under autocast the latent arrives as bf16; the quantizer must
    # run its codebook math in float32 and return the caller's dtype.
    rvq = HierarchicalResidualVQ(RVQConfig(dim=16, n_macro=8, n_micro=8)).train()
    z = torch.randn(32, 16, dtype=torch.bfloat16)
    q, info = rvq(z)
    assert q.dtype == torch.bfloat16
    assert torch.isfinite(info["vq_loss"])


def test_codebook_usage_spreads_after_training():
    torch.manual_seed(1)
    cfg = RVQConfig(dim=8, n_macro=16, n_micro=16, decay=0.9)
    rvq = HierarchicalResidualVQ(cfg).train()
    centers = torch.randn(10, 8) * 4
    used = set()
    for _ in range(80):
        idx = torch.randint(0, 10, (256,))
        data = centers[idx] + 0.05 * torch.randn(256, 8)
        _, info = rvq(data)
        used.update(info["macro_idx"].flatten().tolist())
    assert len(used) >= 5  # multiple macro codes active, not collapsed to one
