"""Isometric (distance-preserving) metric-learning loss + collapse telemetry.

The encoder is trained so that pairwise distances in the 3D OKLab output match the
pairwise distances in the 50-D physical input. Both distance matrices are normalised
to ``[0,1]`` (their maxima differ wildly between 50-D and 3-D space) and compared by
MSE:

    D_in  = ||x_i - x_j||   over the [N,50] state,   D_in  /= D_in.max()
    D_out = ||c_i - c_j||   over the [N,3]  colour,  D_out /= D_out.max()
    L     = mean( (D_out - D_in)^2 )

This corners the network: if a body moves in the 50-D input the physical distances
change, forcing the colour to move to keep the geometry — no bottleneck can "solar
overfit" by ignoring the outer planets, and a collapsed (constant) colour is heavily
penalised because then D_out ~ 0 while D_in is large.

``oklab_stats`` is imported from the root autoencoder as the mode-collapse alarm.
"""

from __future__ import annotations

import torch

from kalachakra.local_autoencoder.losses import oklab_stats    # reuse, don't rewrite

__all__ = ["isometric_loss", "oklab_stats"]


def isometric_loss(state: torch.Tensor, oklab: torch.Tensor,
                   eps: float = 1e-8) -> torch.Tensor:
    """Normalised pairwise-distance MSE between the 50-D state and the 3-D colour."""
    d_in = torch.cdist(state, state)                           # [N,N] physical distances
    d_out = torch.cdist(oklab, oklab)                         # [N,N] colour distances
    d_in = d_in / (d_in.max() + eps)                          # -> [0,1]
    d_out = d_out / (d_out.max() + eps)                       # -> [0,1]
    return ((d_out - d_in) ** 2).mean()
