"""
Geodesic (spherical) convolution over the icosahedral Earth mesh (blueprint §4.2).

A true continuous spherical convolution is approximated here by localized
message passing over precomputed geodesic neighborhoods: each node aggregates a
learned combination of its ``k`` nearest neighbors on the sphere. Because the
neighbor sets are defined by angular (great-circle) proximity, the operator
preserves geodesic relationships without any map projection — satisfying the
non-Euclidean requirement of §2.2 while remaining cheap enough for the M4 Max.

``build_knn`` (pure numpy) precomputes the neighbor index tensor once from a
:class:`~kalachakra.grid.geodesic.Grid`; the layer itself is PyTorch.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..grid.geodesic import Grid


def build_knn(grid: Grid, k: int) -> np.ndarray:
    """Return ``(N, k)`` indices of each node's ``k`` nearest neighbors (incl. self).

    Neighbors are ranked by great-circle distance == descending cosine similarity
    of the unit vectors. For very large meshes this should be chunked or replaced
    with an ANN index; the exact form is kept here for clarity and testing.
    """
    xyz = grid.xyz
    n = xyz.shape[0]
    k = min(k, n)
    idx = np.empty((n, k), dtype=np.int64)
    # Row-chunked to bound peak memory on the full 122,880-node mesh.
    chunk = 2048
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        sims = xyz[start:end] @ xyz.T                  # cosine similarity
        idx[start:end] = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    return idx


class GeodesicConv(nn.Module):
    """Neighborhood message-passing convolution on a fixed spherical mesh.

    Input  : ``(batch, N_nodes, in_channels)``
    Output : ``(batch, N_nodes, out_channels)``

    The neighbor index tensor is registered as a buffer so it moves with the
    module across devices (CPU / MPS) and is saved in checkpoints.
    """

    def __init__(self, in_channels: int, out_channels: int, neighbors: np.ndarray):
        super().__init__()
        self.k = neighbors.shape[1]
        self.register_buffer("neighbors", torch.as_tensor(neighbors, dtype=torch.long))
        # Self weight + shared neighbor weight (isotropic kernel).
        self.self_lin = nn.Linear(in_channels, out_channels)
        self.neigh_lin = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gather neighbor features: (batch, N, k, C_in) then mean-aggregate.
        idx = self.neighbors                                   # (N, k)
        neigh = x[:, idx, :]                                   # (batch, N, k, C_in)
        agg = neigh.mean(dim=2)                                # (batch, N, C_in)
        return self.self_lin(x) + self.neigh_lin(agg)


class SphericalPool(nn.Module):
    """Coarsen the mesh by selecting a subset of nodes and re-mixing features.

    A pragmatic pooling: keep every ``stride``-th node (the mesh is near-uniform,
    so strided selection preserves global coverage) after a neighborhood mix.
    """

    def __init__(self, conv: GeodesicConv, stride: int):
        super().__init__()
        self.conv = conv
        self.stride = stride

    def forward(self, x: torch.Tensor):
        x = torch.relu(self.conv(x))
        return x[:, :: self.stride, :]
