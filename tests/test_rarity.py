import numpy as np

from kalachakra.analysis.rarity import RarityModel


def test_pmf_normalizes():
    m = RarityModel(n_tokens=8)
    m.fit(np.array([0, 0, 1, 2, 2, 2]))
    assert abs(m.pmf.sum() - 1.0) < 1e-9
    assert m.total == 6


def test_rare_token_high_common_token_low():
    m = RarityModel(n_tokens=100)
    # token 0 appears 9,999 times; token 1 appears once.
    idx = np.concatenate([np.zeros(9999, dtype=int), np.array([1])])
    m.fit(idx)
    r = m.rarity(np.array([0, 1]))
    assert r[0] < 0.1          # very common -> ~0
    assert r[1] > 0.9          # seen once in ~10k -> ~1
    assert 0.0 <= r.min() and r.max() <= 1.0


def test_significance_threshold_monotonic_in_percentile():
    m = RarityModel(n_tokens=64)
    rng = np.random.default_rng(0)
    m.fit(rng.integers(0, 64, size=100_000))
    t50 = m.significance_threshold(50)
    t99 = m.significance_threshold(99)
    t9999 = m.significance_threshold(99.99)
    assert t50 <= t99 <= t9999   # tighter percentile -> higher rarity cutoff


def test_unseen_token_gets_finite_rarity():
    m = RarityModel(n_tokens=16)
    m.fit(np.array([0, 0, 1]))
    r = m.rarity(np.array([15]))   # never seen
    assert np.isfinite(r).all() and r[0] <= 1.0


def test_percentile_threshold_of_distribution():
    m = RarityModel(n_tokens=8)
    vals = np.linspace(0, 1, 101)
    assert abs(m.percentile_threshold(vals, 90) - 0.9) < 1e-6
