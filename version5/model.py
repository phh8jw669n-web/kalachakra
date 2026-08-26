"""The Sky-Energy Autoencoder.

Encoder: each body's five local angles ``[alt, az, RA, dec, HA]`` are expanded to
``(sin, cos)`` (so 0/2pi wraps are seamless), projected to ``d_model``, joined by a
learnable ``<OBSERVER>`` token, and mixed by a self-attention Transformer — the
attention is permutation-invariant, so the planets have no artificial "order" and
the model is free to discover geometric aspects (trines, squares, oppositions).

Bottleneck: the observer token is funnelled through a **3-neuron** linear layer that
*is* the OKLab colour — ``L`` via sigmoid ``[0,1]``, ``a`` and ``b`` via tanh
``[-1,1]`` (PRD page 3). There is no hidden dimension to bypass it.

Decoder (training only): an MLP ``3 -> 64 -> 256 -> 10*4`` reconstructs the
``(sin,cos)`` of every planet's altitude and azimuth. If the loss falls, the three
colours provably contain the entire local geometry.

The Transformer block is **imported** from the root package, never re-implemented.
"""

from __future__ import annotations

import torch
from torch import nn

from kalachakra.local_autoencoder.model import AttentionEncoderLayer  # reuse, don't rewrite

from .config import ModelConfig


class SkyEnergyEncoder(nn.Module):
    """``[B,10,5]`` local angles -> OKLab ``[B,3]`` (the exported half)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.input_proj = nn.Linear(2 * cfg.raw_features, d)     # (sin,cos) -> d
        self.input_norm = nn.LayerNorm(d)
        self.observer_token = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.observer_token, std=0.02)
        self.layers = nn.ModuleList([
            AttentionEncoderLayer(d, cfg.nhead, cfg.dim_feedforward, cfg.dropout)
            for _ in range(cfg.num_layers)
        ])
        self.to_bottleneck = nn.Linear(d, 3)

    @staticmethod
    def _expand_angles(feats: torch.Tensor) -> torch.Tensor:
        """``[B,10,5]`` radians -> ``[B,10,10]`` = ``[sin(f), cos(f)]``."""
        return torch.cat([torch.sin(feats), torch.cos(feats)], dim=-1)

    @staticmethod
    def _bound_oklab(h: torch.Tensor) -> torch.Tensor:
        L = torch.sigmoid(h[..., :1])            # [0,1]
        ab = torch.tanh(h[..., 1:])              # [-1,1]
        return torch.cat([L, ab], dim=-1)

    def forward(self, feats: torch.Tensor, return_attention: bool = False):
        b = feats.shape[0]
        tok = self.input_norm(self.input_proj(self._expand_angles(feats)))   # [B,10,d]
        obs = self.observer_token.expand(b, -1, -1)                          # [B,1,d]
        seq = torch.cat([tok, obs], dim=1)                                   # [B,11,d]
        attns = []
        for layer in self.layers:
            seq, attn = layer(seq, need_weights=return_attention)
            if return_attention:
                attns.append(attn)
        pooled = seq[:, -1] if self.cfg.pool == "observer" else seq.mean(dim=1)
        oklab = self._bound_oklab(self.to_bottleneck(pooled))               # [B,3]
        if return_attention:
            return oklab, torch.stack(attns, dim=1)                         # [B,L,H,11,11]
        return oklab


class SkyEnergyAutoencoder(nn.Module):
    """Encoder + decoder. ``forward -> (recon [B,10,4], oklab [B,3])``."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = SkyEnergyEncoder(cfg)
        out = cfg.n_bodies * cfg.recon_features                             # 40
        dims = [3, *cfg.decoder_hidden, out]
        dec: list[nn.Module] = []
        for i in range(len(dims) - 1):
            dec.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                dec.append(nn.GELU())
        self.decoder = nn.Sequential(*dec)

    def encode(self, feats: torch.Tensor, return_attention: bool = False):
        return self.encoder(feats, return_attention=return_attention)

    def forward(self, feats: torch.Tensor):
        oklab = self.encoder(feats)
        recon = self.decoder(oklab).view(-1, self.cfg.n_bodies, self.cfg.recon_features)
        return recon, oklab


def build_model(cfg: ModelConfig) -> SkyEnergyAutoencoder:
    return SkyEnergyAutoencoder(cfg)
