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
from .autoencoder_v2 import DEFAULT_NODE_CHUNK, STBlockV2, SphericalAutoencoderV2
from .quantized_autoencoder import QuantizedSphericalAutoencoder
from .rvq import RVQConfig


class QuantizedSphericalAutoencoderV2(QuantizedSphericalAutoencoder):
    """Quantized autoencoder with node-chunked forward (large-mesh / MPS safe).

    Optional ``grad_checkpoint`` recomputes the encoder/decoder block activations
    in the backward pass to fit the full mesh in memory. The RVQ sits between
    ``ae.encode`` and ``ae.decode`` — outside the checkpointed blocks — so its EMA
    codebook update still runs exactly once per step.
    """

    def __init__(self, ae_cfg: AutoencoderConfig, neighbors: np.ndarray,
                 rvq_cfg: RVQConfig | None = None,
                 node_chunk: int = DEFAULT_NODE_CHUNK,
                 grad_checkpoint: bool = False):
        super().__init__(ae_cfg, neighbors, rvq_cfg)
        # Give the inner AE the v2 chunked/checkpointing encode+decode (state_dict
        # keys under ``ae.*`` are unchanged, so checkpoints stay interchangeable).
        self.ae.__class__ = SphericalAutoencoderV2
        self.ae._grad_checkpoint = grad_checkpoint
        for blk in list(self.ae.enc_blocks) + list(self.ae.dec_blocks):
            blk.__class__ = STBlockV2
            blk.node_chunk = node_chunk
