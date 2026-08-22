"""
Hierarchical Spherical Autoencoder + Spatio-Temporal FNO (blueprint §4).

The encoder mixes information across space (geodesic convolutions on the mesh)
and time (Fourier layers on the temporal axis) and projects each spatio-temporal
coordinate to a 64-dimensional latent code z(t, s). The decoder is a mirror image
that reconstructs the 50-dimensional local field E(t, s). Because each latent
code summarizes a *window* of frames and a *neighborhood* of nodes, the fixed
64-d width is a genuine bottleneck relative to the receptive volume, forcing the
network to encode geometric invariants rather than memorize samples (§4.3).

Tensor convention throughout::

    E : (batch, T, N, F)      F == LOCAL_FIELD_WIDTH (50)
    z : (batch, T, N, LATENT) LATENT == 64

Requires PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .. import constants as C
from .fno import FourierBlock1d
from .spherical_conv import GeodesicConv


@dataclass
class AutoencoderConfig:
    n_nodes: int = C.N_SPATIAL_NODES
    in_features: int = C.LOCAL_FIELD_WIDTH      # 50
    hidden: int = 128
    latent: int = C.LATENT_DIM                  # 64
    fourier_modes: int = 32
    knn: int = 7
    n_blocks: int = 3


def _apply_temporal(block: FourierBlock1d, x: torch.Tensor) -> torch.Tensor:
    """Apply a (channels-first, time-last) FNO block to ``(B, T, N, Ch)`` data."""
    b, t, n, ch = x.shape
    # -> (B*N, Ch, T) so the FFT runs along the temporal axis.
    y = x.permute(0, 2, 3, 1).reshape(b * n, ch, t)
    y = block(y)
    return y.reshape(b, n, ch, t).permute(0, 3, 1, 2).contiguous()


def _apply_spatial(conv: GeodesicConv, x: torch.Tensor) -> torch.Tensor:
    """Apply a geodesic conv to ``(B, T, N, Ch)`` by folding time into the batch."""
    b, t, n, ch = x.shape
    y = x.reshape(b * t, n, ch)
    y = conv(y)
    return y.reshape(b, t, n, -1)


class STBlock(nn.Module):
    """One spatio-temporal block: spatial geodesic conv then temporal FNO."""

    def __init__(self, channels: int, modes: int, neighbors: np.ndarray):
        super().__init__()
        self.spatial = GeodesicConv(channels, channels, neighbors)
        self.temporal = FourierBlock1d(channels, modes)
        self.norm = nn.LayerNorm(channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.act(_apply_spatial(self.spatial, x))
        x = x + _apply_temporal(self.temporal, x)
        return self.norm(x)


class SphericalAutoencoder(nn.Module):
    """Encoder + 64-d latent bottleneck + decoder over the local field E(t, s)."""

    def __init__(self, cfg: AutoencoderConfig, neighbors: np.ndarray):
        super().__init__()
        self.cfg = cfg
        self.lift = nn.Linear(cfg.in_features, cfg.hidden)
        self.enc_blocks = nn.ModuleList(
            STBlock(cfg.hidden, cfg.fourier_modes, neighbors)
            for _ in range(cfg.n_blocks)
        )
        self.to_latent = nn.Linear(cfg.hidden, cfg.latent)

        self.from_latent = nn.Linear(cfg.latent, cfg.hidden)
        self.dec_blocks = nn.ModuleList(
            STBlock(cfg.hidden, cfg.fourier_modes, neighbors)
            for _ in range(cfg.n_blocks)
        )
        self.project = nn.Linear(cfg.hidden, cfg.in_features)

    def encode(self, e: torch.Tensor) -> torch.Tensor:
        x = self.lift(e)
        for block in self.enc_blocks:
            x = block(x)
        return self.to_latent(x)                      # (B, T, N, latent)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.from_latent(z)
        for block in self.dec_blocks:
            x = block(x)
        return self.project(x)                        # (B, T, N, in_features)

    def forward(self, e: torch.Tensor):
        z = self.encode(e)
        recon = self.decode(z)
        return recon, z
