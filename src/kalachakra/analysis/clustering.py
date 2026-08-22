"""
Manifold clustering engine (blueprint §6.2).

Density-based clustering (HDBSCAN) over the stream of latent vectors groups
recurring waveform-interference patterns into stable clusters — high-tension
orthogonal collisions in one region of the manifold, harmonious trine resonances
in another — with no predefined categories.

``hdbscan`` is optional; when it is not installed a deterministic, dependency-free
fallback (normalized-cut style greedy density grouping) keeps the API usable for
smoke tests. The fallback is clearly labeled and not a substitute for HDBSCAN at
scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # pragma: no cover - optional dependency
    import hdbscan as _hdbscan

    _HAS_HDBSCAN = True
except Exception:  # noqa: BLE001
    _hdbscan = None
    _HAS_HDBSCAN = False


@dataclass
class ClusterResult:
    labels: np.ndarray            # (M,) int, -1 == noise
    n_clusters: int
    method: str


def cluster_latents(z: np.ndarray, *, min_cluster_size: int = 50,
                    min_samples: int | None = None) -> ClusterResult:
    """Cluster flattened latent vectors ``z`` of shape ``(M, LATENT)``."""
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError("z must be 2D (M, LATENT); flatten (t,s) first")

    if _HAS_HDBSCAN:
        clusterer = _hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size, min_samples=min_samples
        )
        labels = clusterer.fit_predict(z)
        n = int(labels.max()) + 1 if labels.size and labels.max() >= 0 else 0
        return ClusterResult(labels=labels, n_clusters=n, method="hdbscan")

    return _fallback_cluster(z, min_cluster_size)


def _fallback_cluster(z: np.ndarray, min_cluster_size: int) -> ClusterResult:
    """Greedy radius grouping around density peaks (fallback only)."""
    m = z.shape[0]
    labels = np.full(m, -1, dtype=np.int64)
    if m == 0:
        return ClusterResult(labels=labels, n_clusters=0, method="fallback")

    # Characteristic scale: median nearest-neighbor-ish distance via sampling.
    sample = z[np.random.default_rng(0).choice(m, min(m, 256), replace=False)]
    dists = np.linalg.norm(sample[:, None] - sample[None], axis=-1)
    np.fill_diagonal(dists, np.inf)
    radius = float(np.median(dists.min(axis=1))) * 2.0 or 1.0

    cluster_id = 0
    for i in range(m):
        if labels[i] != -1:
            continue
        d = np.linalg.norm(z - z[i], axis=1)
        members = np.where(d <= radius)[0]
        if members.size >= min_cluster_size:
            labels[members[labels[members] == -1]] = cluster_id
            cluster_id += 1
    return ClusterResult(labels=labels, n_clusters=cluster_id, method="fallback")


def hdbscan_available() -> bool:
    return _HAS_HDBSCAN
