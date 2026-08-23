"""
Three-tier temporal mipmapping (blueprint §3).

To keep any query bounded to ~1000 processed rows regardless of span:

* **Tier 1** — native 24-second frames (full fidelity, minutes-to-days windows).
* **Tier 2** — hourly rollups: one bucket per 150 frames (150 x 24 s = 1 h),
  recording max geometric potential, peak spatial shear, and the active VQ
  archetype (modal leaf token) in the bucket.
* **Tier 3** — daily / epochal rollups: means, maxima, spreads, modal archetype,
  and a structural-anomaly count (frames whose rarity exceeds a threshold).

Pure numpy; fully tested. The reductions handle a ragged final bucket.
"""

from __future__ import annotations

import numpy as np

FRAMES_PER_HOUR = 150        # 150 * 24 s == 3600 s
FRAMES_PER_DAY = 3600        # 3600 * 24 s == 86400 s


def _bucket_starts(n: int, bucket: int) -> np.ndarray:
    if bucket <= 0:
        raise ValueError("bucket must be positive")
    return np.arange(0, n, bucket)


def _bucket_counts(n: int, bucket: int) -> np.ndarray:
    starts = _bucket_starts(n, bucket)
    ends = np.append(starts[1:], n)
    return ends - starts


def bucket_max(values: np.ndarray, bucket: int) -> np.ndarray:
    """Per-bucket maximum along axis 0 (handles a ragged final bucket)."""
    values = np.asarray(values, dtype=np.float64)
    return np.maximum.reduceat(values, _bucket_starts(values.shape[0], bucket), axis=0)


def bucket_mean(values: np.ndarray, bucket: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    starts = _bucket_starts(values.shape[0], bucket)
    counts = _bucket_counts(values.shape[0], bucket)
    s = np.add.reduceat(values, starts, axis=0)
    shape = [1] * values.ndim
    shape[0] = counts.shape[0]
    return s / counts.reshape(shape)


def bucket_std(values: np.ndarray, bucket: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    mean = bucket_mean(values, bucket)
    mean_sq = bucket_mean(values * values, bucket)
    return np.sqrt(np.clip(mean_sq - mean * mean, 0.0, None))


def mode_per_bucket(tokens: np.ndarray, bucket: int, n_tokens: int) -> np.ndarray:
    """Most frequent token id in each bucket (the 'active archetype')."""
    tokens = np.asarray(tokens, dtype=np.int64)
    starts = _bucket_starts(tokens.shape[0], bucket)
    ends = np.append(starts[1:], tokens.shape[0])
    out = np.empty(starts.shape[0], dtype=np.int64)
    for i, (a, b) in enumerate(zip(starts, ends)):
        out[i] = np.bincount(tokens[a:b], minlength=n_tokens).argmax()
    return out


def hourly_rollup(potential: np.ndarray, shear: np.ndarray, leaf: np.ndarray,
                  n_tokens: int = 4096) -> dict[str, np.ndarray]:
    """Tier 2 — hourly (150-frame) rollup of per-frame scalars/tokens."""
    return {
        "max_potential": bucket_max(potential, FRAMES_PER_HOUR),
        "peak_shear": bucket_max(shear, FRAMES_PER_HOUR),
        "archetype": mode_per_bucket(leaf, FRAMES_PER_HOUR, n_tokens),
        "n_buckets": _bucket_starts(len(potential), FRAMES_PER_HOUR).shape[0],
    }


def daily_rollup(potential: np.ndarray, shear: np.ndarray, rarity: np.ndarray,
                 leaf: np.ndarray, rarity_threshold: float = 0.9,
                 n_tokens: int = 4096) -> dict[str, np.ndarray]:
    """Tier 3 — daily (3600-frame) statistical rollup + anomaly flags."""
    b = FRAMES_PER_DAY
    rarity = np.asarray(rarity, dtype=np.float64)
    starts = _bucket_starts(rarity.shape[0], b)
    anomaly = np.add.reduceat((rarity >= rarity_threshold).astype(np.float64),
                              starts, axis=0)
    return {
        "mean_potential": bucket_mean(potential, b),
        "max_potential": bucket_max(potential, b),
        "std_potential": bucket_std(potential, b),
        "peak_shear": bucket_max(shear, b),
        "mean_rarity": bucket_mean(rarity, b),
        "max_rarity": bucket_max(rarity, b),
        "anomaly_count": anomaly,
        "archetype": mode_per_bucket(leaf, b, n_tokens),
    }


def select_tier(span_frames: int, target_rows: int = 1000) -> str:
    """Pick the coarsest tier that keeps a scan under ``target_rows`` rows.

    Returns "tier1" (native), "tier2" (hourly), or "tier3" (daily).
    """
    if span_frames <= target_rows:
        return "tier1"
    if span_frames / FRAMES_PER_HOUR <= target_rows:
        return "tier2"
    return "tier3"
