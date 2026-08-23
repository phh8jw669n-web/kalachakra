"""
Spherical Autoencoder + Hierarchical Residual VQ (blueprint §1-§2).

Inserts the two-level quantizer between the encoder and decoder (VQ-VAE style):

    E(t,s) --encode--> z(64) --RVQ--> quantized(64) --decode--> reconstruction

The decoder reconstructs from the *quantized* latent, so the discrete tokens must
carry enough geometry to rebuild the field. Training minimizes reconstruction +
the RVQ commitment loss; the straight-through estimator carries gradients through
the quantizer to the encoder.

Requires PyTorch.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .autoencoder import AutoencoderConfig, SphericalAutoencoder
from .rvq import HierarchicalResidualVQ, RVQConfig


class QuantizedSphericalAutoencoder(nn.Module):
    """Autoencoder whose latent is discretized by a hierarchical residual VQ."""

    def __init__(self, ae_cfg: AutoencoderConfig, neighbors: np.ndarray,
                 rvq_cfg: RVQConfig | None = None):
        super().__init__()
        self.ae = SphericalAutoencoder(ae_cfg, neighbors)
        self.rvq = HierarchicalResidualVQ(rvq_cfg or RVQConfig(dim=ae_cfg.latent))

    def encode_quantize(self, e: torch.Tensor):
        """Return (continuous z, quantized z, vq info) without decoding."""
        z = self.ae.encode(e)
        quantized, info = self.rvq(z)
        return z, quantized, info

    def forward(self, e: torch.Tensor):
        z = self.ae.encode(e)
        quantized, info = self.rvq(z)
        recon = self.ae.decode(quantized)
        return recon, z, quantized, info

    @torch.no_grad()
    def tokenize(self, e: torch.Tensor):
        """Inference: return (macro_idx, micro_idx, leaf_idx, quantized latent)."""
        self.eval()
        z = self.ae.encode(e)
        quantized, info = self.rvq(z)
        return info["macro_idx"], info["micro_idx"], info["leaf_idx"], quantized
