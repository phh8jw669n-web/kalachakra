"""The Sky-Energy metric encoder (version5.1).

A pure forward-pass encoder — no decoder. It maps the Zero-Redundancy 50-D physical
state to a 3D OKLab colour, trained by a distance-preserving (isometric) loss rather
than reconstruction, so pairwise colour distances match pairwise physical distances.

The 50-D state is read as a sequence of tokens: 11 body tokens (each already a 3D
ecliptic Cartesian unit vector + normalised velocity) and 1 observer token (Ascendant
+ Midheaven Cartesian). Because the inputs are already bounded Cartesian coordinates
there is no sin/cos expansion — each token is projected straight to ``d_model``. The
self-attention Transformer block is imported from the root package; the observer token
is pooled and projected to 3 OKLab neurons (``L`` sigmoid ``[0,1]``, ``a,b`` tanh
``[-1,1]``) so the output stays inside the colour gamut.
"""

from __future__ import annotations

import torch
from torch import nn

from kalachakra.local_autoencoder.model import AttentionEncoderLayer  # reuse, don't rewrite

from .config import ModelConfig


class SkyEnergyEncoder(nn.Module):
    """``state [N,50] -> OKLab [N,3]``. This *is* the whole model (encoder only)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.body_dim = cfg.n_bodies * cfg.body_features          # 44
        self.body_proj = nn.Linear(cfg.body_features, d)          # 4 -> d (per body token)
        self.body_norm = nn.LayerNorm(d)
        self.observer_proj = nn.Linear(cfg.obs_features, d)       # 6 -> d (observer token)
        self.observer_norm = nn.LayerNorm(d)
        self.layers = nn.ModuleList([
            AttentionEncoderLayer(d, cfg.nhead, cfg.dim_feedforward, cfg.dropout)
            for _ in range(cfg.num_layers)
        ])
        self.to_bottleneck = nn.Linear(d, 3)

    @staticmethod
    def _bound_oklab(h: torch.Tensor) -> torch.Tensor:
        L = torch.sigmoid(h[..., :1])                             # [0,1]
        ab = torch.tanh(h[..., 1:])                               # [-1,1]
        return torch.cat([L, ab], dim=-1)

    def forward(self, state: torch.Tensor, return_attention: bool = False):
        b = state.shape[0]
        bodies = state[:, :self.body_dim].view(b, self.cfg.n_bodies, self.cfg.body_features)
        observer = state[:, self.body_dim:]                       # [B,6]
        tok = self.body_norm(self.body_proj(bodies))              # [B,11,d]
        obs = self.observer_norm(self.observer_proj(observer)).unsqueeze(1)   # [B,1,d]
        seq = torch.cat([tok, obs], dim=1)                        # [B,12,d]  obs = idx 11
        attns = []
        for layer in self.layers:
            seq, attn = layer(seq, need_weights=return_attention)
            if return_attention:
                attns.append(attn)
        pooled = seq[:, -1] if self.cfg.pool == "observer" else seq.mean(dim=1)
        oklab = self._bound_oklab(self.to_bottleneck(pooled))     # [B,3]
        if return_attention:
            return oklab, torch.stack(attns, dim=1)               # [B,L,H,12,12]
        return oklab


def build_model(cfg: ModelConfig) -> SkyEnergyEncoder:
    return SkyEnergyEncoder(cfg)
