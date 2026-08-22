"""
Anomaly and singularity detection (blueprint §6.3).

Statistical thresholding over the geometric potential field and temporal shear
gradient isolates rare, high-amplitude convergence events. A *singularity* is a
spatio-temporal coordinate where both fields exceed their high-percentile
thresholds simultaneously — multiple heavy waveforms compressing into one
orthogonal focal point.

Pure numpy and fully tested.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Singularity:
    time_index: int
    node_index: int
    potential: float
    shear: float
    score: float


def robust_threshold(field: np.ndarray, sigma: float = 4.0) -> float:
    """Median + ``sigma`` * MAD threshold (robust to the heavy tail we hunt for)."""
    field = np.asarray(field, dtype=np.float64)
    median = np.median(field)
    mad = np.median(np.abs(field - median))
    # 1.4826 scales MAD to a std-equivalent for a normal distribution.
    return float(median + sigma * 1.4826 * mad)


def detect_singularities(potential: np.ndarray, shear: np.ndarray, *,
                         sigma: float = 4.0,
                         max_events: int | None = None) -> list[Singularity]:
    """Return coordinates where potential and shear both exceed threshold.

    Both fields have shape ``(T, N)``. Events are ranked by a combined z-score so
    the strongest structural-tension focal points come first.
    """
    potential = np.asarray(potential, dtype=np.float64)
    shear = np.asarray(shear, dtype=np.float64)
    if potential.shape != shear.shape:
        raise ValueError("potential and shear must share shape (T, N)")

    p_thr = robust_threshold(potential, sigma)
    s_thr = robust_threshold(shear, sigma)
    mask = (potential >= p_thr) & (shear >= s_thr)

    def z(x, m):
        std = x.std() or 1.0
        return (m - x.mean()) / std

    events: list[Singularity] = []
    for ti, ni in zip(*np.where(mask)):
        score = float(z(potential, potential[ti, ni]) + z(shear, shear[ti, ni]))
        events.append(Singularity(int(ti), int(ni),
                                   float(potential[ti, ni]),
                                   float(shear[ti, ni]), score))
    events.sort(key=lambda s: s.score, reverse=True)
    if max_events is not None:
        events = events[:max_events]
    return events
