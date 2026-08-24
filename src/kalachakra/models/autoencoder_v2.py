"""
Node-chunked Spatio-Temporal Autoencoder (v2) — MPS/large-mesh safe.

Identical mathematics and parameters to :mod:`kalachakra.models.autoencoder`; the
ONLY difference is that the spatial geodesic conv and the temporal FNO are applied
in slices over the node axis. At the full 122,880-node mesh the v1 forward folds
``batch × nodes`` into one dimension, producing single tensors with more than
INT_MAX (2^31-1) elements — which Apple's Metal (MPS) backend cannot index
("MPSGraph does not support tensor dims larger than INT_MAX"), and which also
materialize ~50+ GB transients. Chunking over nodes keeps every op's tensor small
while producing a numerically equivalent result — the mathematics is identical
(nodes are independent within both the temporal FFT and the neighbor aggregation);
continuous outputs match v1 to float32 precision (any <=~1e-6 residual is ordinary
FFT/matmul batch-tiling non-associativity, far below bf16 training noise) and the
discrete tokens are exact. Model capacity, the 64-d latent, and reconstruction
quality are completely unchanged.

Because the module layout and attribute names match v1 exactly, the state_dict is
interchangeable: a v2-trained checkpoint loads with the v1 loaders and vice versa.

Requires PyTorch.
"""

from __future__ import annotations

import numpy as np
import torch

from .autoencoder import (
    AutoencoderConfig,
    STBlock,
    SphericalAutoencoder,
    _apply_spatial,
    _apply_temporal,
)
from .fno import FourierBlock1d
from .spherical_conv import GeodesicConv

#: Default nodes processed per slice. Sized so each op's tensor stays far under
#: INT_MAX and a few hundred MB: at batch 4 / window 64 / hidden 128 the spatial
#: gather peaks at ~batch*window*chunk*k*hidden and the temporal FFT at
#: ~batch*chunk*hidden*window. Raise for speed, lower if memory is tight.
DEFAULT_NODE_CHUNK = 4096


def _apply_temporal_chunked(block: FourierBlock1d, x: torch.Tensor,
                            node_chunk: int) -> torch.Tensor:
    """Temporal FNO over ``(B, T, N, Ch)`` applied in node slices.

    Each ``(batch, node)`` row is transformed independently along time, so slicing
    the node axis and concatenating is mathematically equal to the un-chunked op
    (up to float32 FFT batch-tiling rounding, ~1e-6).
    """
    b, t, n, ch = x.shape
    if n <= node_chunk:
        return _apply_temporal(block, x)
    outs = []
    for s in range(0, n, node_chunk):
        xs = x[:, :, s:s + node_chunk, :]                     # (b, t, ns, ch)
        ns = xs.shape[2]
        ys = xs.permute(0, 2, 3, 1).reshape(b * ns, ch, t)    # (b*ns, ch, t)
        ys = block(ys)
        outs.append(ys.reshape(b, ns, ch, t).permute(0, 3, 1, 2))  # (b, t, ns, ch)
    return torch.cat(outs, dim=2).contiguous()


def _apply_spatial_chunked(conv: GeodesicConv, x: torch.Tensor,
                           node_chunk: int) -> torch.Tensor:
    """Geodesic conv over ``(B, T, N, Ch)`` with the neighbor gather done in node
    slices. Each output node depends only on its own ``k`` neighbors, so the
    chunked aggregation is exactly equal to the full one; the big transient is the
    ``(B*T, N, k, Ch)`` gather, which chunking bounds to one slice at a time.
    """
    b, t, n, ch = x.shape
    if n <= node_chunk:
        return _apply_spatial(conv, x)
    xf = x.reshape(b * t, n, ch)
    idx = conv.neighbors                                       # (N, k)
    aggs = []
    for s in range(0, n, node_chunk):
        nb = idx[s:s + node_chunk]                            # (ns, k)
        aggs.append(xf[:, nb, :].mean(dim=2))                # (b*t, ns, ch)
    agg = torch.cat(aggs, dim=1)                              # (b*t, n, ch)
    y = conv.self_lin(xf) + conv.neigh_lin(agg)              # (b*t, n, out)
    return y.reshape(b, t, n, -1)


class STBlockV2(STBlock):
    """STBlock with node-chunked spatial + temporal application (same params)."""

    #: Set per-instance by the autoencoder; class default keeps it usable alone.
    node_chunk: int = DEFAULT_NODE_CHUNK

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.act(_apply_spatial_chunked(self.spatial, x, self.node_chunk))
        x = x + _apply_temporal_chunked(self.temporal, x, self.node_chunk)
        return self.norm(x)


class SphericalAutoencoderV2(SphericalAutoencoder):
    """Autoencoder whose STBlocks apply their ops in node slices (MPS/large-mesh
    safe). Weights and state_dict are identical to :class:`SphericalAutoencoder`."""

    def __init__(self, cfg: AutoencoderConfig, neighbors: np.ndarray,
                 node_chunk: int = DEFAULT_NODE_CHUNK):
        super().__init__(cfg, neighbors)
        # Retarget the already-built blocks to the chunked forward. Reassigning
        # __class__ keeps every parameter/buffer (and thus every state_dict key)
        # exactly as v1 built it — only the forward computation changes.
        for blk in list(self.enc_blocks) + list(self.dec_blocks):
            blk.__class__ = STBlockV2
            blk.node_chunk = node_chunk
