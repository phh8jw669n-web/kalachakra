"""
Deep-time statistical rarity from the discrete token stream (blueprint §2, §6).

Across the whole 13.4-billion-frame corpus the system maintains an empirical
probability mass function over the 4096 leaf tokens. The **Rarity Index** of a
configuration is the normalized negative log-likelihood of its assigned token:
common configurations -> ~0, configurations seen only a handful of times across
ten millennia -> ~1. This is an unbiased anomaly metric grounded purely in
statistical uniqueness, with no astrological convention.

Pure numpy; fully tested.
"""

from __future__ import annotations

import numpy as np


class RarityModel:
    """Empirical token PMF + normalized-NLL rarity over a fixed token alphabet."""

    def __init__(self, n_tokens: int = 4096):
        self.n_tokens = int(n_tokens)
        self.counts = np.zeros(self.n_tokens, dtype=np.float64)

    # -- fitting ----------------------------------------------------------
    def fit(self, token_indices: np.ndarray) -> "RarityModel":
        """Reset and count the token histogram from a full corpus pass."""
        self.counts[:] = 0.0
        return self.update(token_indices)

    def update(self, token_indices: np.ndarray) -> "RarityModel":
        """Incrementally add counts (for streaming the corpus in chunks)."""
        idx = np.asarray(token_indices, dtype=np.int64).ravel()
        if idx.size:
            if idx.min() < 0 or idx.max() >= self.n_tokens:
                raise ValueError("token index out of range")
            self.counts += np.bincount(idx, minlength=self.n_tokens)
        return self

    @property
    def total(self) -> float:
        return float(self.counts.sum())

    @property
    def pmf(self) -> np.ndarray:
        t = self.total
        return self.counts / t if t else np.full(self.n_tokens, 1.0 / self.n_tokens)

    # -- rarity -----------------------------------------------------------
    def nll(self, token_indices: np.ndarray) -> np.ndarray:
        """Negative log-likelihood (nats) of each token under the PMF."""
        idx = np.asarray(token_indices, dtype=np.int64)
        p = self.pmf[idx]
        # Unseen tokens get the add-one-smoothed floor rather than +inf.
        floor = 1.0 / (self.total + self.n_tokens) if self.total else 1.0 / self.n_tokens
        p = np.where(p > 0.0, p, floor)
        return -np.log(p)

    def rarity(self, token_indices: np.ndarray) -> np.ndarray:
        """Rarity Index in [0, 1]: NLL normalized by the max possible NLL.

        The rarest observable event (seen once in ``total`` frames) has
        ``NLL = log(total)``, which maps to 1.0; frequent tokens map toward 0.
        """
        denom = np.log(max(self.total, self.n_tokens))
        r = self.nll(token_indices) / denom if denom > 0 else self.nll(token_indices)
        return np.clip(r, 0.0, 1.0)

    # -- thresholds (for the dynamic news radar, §6) ----------------------
    def percentile_threshold(self, rarities: np.ndarray, q: float) -> float:
        """Rarity value at percentile ``q`` (0-100) of an observed distribution."""
        return float(np.percentile(np.asarray(rarities, dtype=np.float64), q))

    def significance_threshold(self, q: float) -> float:
        """Rarity threshold at percentile ``q`` of the *token* rarity spectrum.

        Uses each token's own rarity weighted by how often it occurs, so the
        threshold reflects the frame-level distribution (used by §6 to tighten
        toward the 99.99th percentile at deep-time zoom).
        """
        all_tokens = np.arange(self.n_tokens)
        token_rarity = self.rarity(all_tokens)
        # Expand by counts so frequent tokens weigh proportionally.
        weights = self.counts
        if weights.sum() == 0:
            return float(np.percentile(token_rarity, q))
        order = np.argsort(token_rarity)
        sorted_r = token_rarity[order]
        cum = np.cumsum(weights[order])
        cutoff = (q / 100.0) * cum[-1]
        i = int(np.searchsorted(cum, cutoff))
        return float(sorted_r[min(i, self.n_tokens - 1)])
