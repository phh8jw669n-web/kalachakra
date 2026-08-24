"""
Node-chunked Quantized Spherical Autoencoder (v2) — MPS/large-mesh safe.

Same as :class:`kalachakra.models.quantized_autoencoder.QuantizedSphericalAutoencoder`
(encoder + hierarchical residual VQ + decoder), but the inner autoencoder applies
its spatial/temporal ops in node slices so the full 122,880-node mesh never builds
a tensor larger than Metal's INT_MAX limit. Mathematics, parameters, tokens, and
state_dict keys are identical to v1 — only the forward tiling changes.

Requires PyTorch.
"""

from __future__ import annotations

import numpy as np

from .autoencoder import AutoencoderConfig
from .autoencoder_v2 import DEFAULT_NODE_CHUNK, STBlockV2
from .quantized_autoencoder import QuantizedSphericalAutoencoder
from .rvq import RVQConfig


class QuantizedSphericalAutoencoderV2(QuantizedSphericalAutoencoder):
    """Quantized autoencoder with node-chunked forward (large-mesh / MPS safe)."""

    def __init__(self, ae_cfg: AutoencoderConfig, neighbors: np.ndarray,
                 rvq_cfg: RVQConfig | None = None,
                 node_chunk: int = DEFAULT_NODE_CHUNK):
        super().__init__(ae_cfg, neighbors, rvq_cfg)
        for blk in list(self.ae.enc_blocks) + list(self.ae.dec_blocks):
            blk.__class__ = STBlockV2
            blk.node_chunk = node_chunk
