"""The Local Sky Autoencoder: (10,8) physics -> OKLab(3) bottleneck -> (11,8).

Encoder: project each body's features (cyclic angles expanded to ``(sin,cos)``) to
``d_model``, append a trainable ``<OBSERVER>`` token, run a multi-head self-attention
Transformer (so the network computes geometric interference), pool the observer
token (or global-average), and project to 3 OKLab neurons bounded to
``L in [0,1]``, ``a,b in [-0.5,0.5]``.

Decoder: an expanding MLP ``3 -> 64 -> 256 -> 11*8`` reshaped to ``(11,8)`` that
reconstructs the Local Sky Matrix (+ the observer row).
"""

from __future__ import annotations

import torch
from torch import nn

from .config import ModelConfig
from .features import ANGULAR_COLS, BODY_NAMES, N_TOKENS, SCALAR_COLS


class AttentionEncoderLayer(nn.Module):
    """Pre-norm Transformer block whose attention weights can be returned."""

    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim_ff, d_model), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, need_weights: bool = False):
        h = self.norm1(x)
        a, attn = self.attn(h, h, h, need_weights=need_weights,
                            average_attn_weights=False)
        x = x + a
        x = x + self.ff(self.norm2(x))
        return x, attn                      # attn: (B, nhead, S, S) or None


class LocalSkyAutoencoder(nn.Module):
    """Physics ``(B,10,8)`` -> OKLab ``(B,3)`` -> reconstruction ``(B,11,8)``."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.register_buffer("_ang", torch.tensor(ANGULAR_COLS, dtype=torch.long))
        self.register_buffer("_sca", torch.tensor(SCALAR_COLS, dtype=torch.long))
        enc_in = 2 * len(ANGULAR_COLS) + len(SCALAR_COLS)             # 13
        self.input_proj = nn.Linear(enc_in, d)
        self.input_norm = nn.LayerNorm(d)
        self.observer_token = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.observer_token, std=0.02)
        self.layers = nn.ModuleList([
            AttentionEncoderLayer(d, cfg.nhead, cfg.dim_feedforward, cfg.dropout)
            for _ in range(cfg.num_layers)
        ])
        self.to_bottleneck = nn.Linear(d, 3)                         # OKLab (L,a,b)

        # decoder: 3 -> hidden... -> 11*8
        dims = [3, *cfg.decoder_hidden, N_TOKENS * cfg.raw_features]
        dec: list[nn.Module] = []
        for i in range(len(dims) - 1):
            dec.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                dec.append(nn.GELU())
        self.decoder = nn.Sequential(*dec)

    # -- encoder -------------------------------------------------------------
    def _expand_angles(self, feats: torch.Tensor) -> torch.Tensor:
        """``(B,10,8)`` -> ``(B,10,13)``: cyclic angles -> (sin,cos), scalars kept."""
        ang = feats.index_select(-1, self._ang)                      # (B,10,A)
        sca = feats.index_select(-1, self._sca)                      # (B,10,S)
        return torch.cat([torch.sin(ang), torch.cos(ang), sca], dim=-1)

    def _bound_oklab(self, h: torch.Tensor) -> torch.Tensor:
        L = 0.5 * (torch.tanh(h[..., :1]) + 1.0)                     # [0,1]
        ab = 0.5 * torch.tanh(h[..., 1:])                           # [-0.5,0.5]
        return torch.cat([L, ab], dim=-1)

    def encode(self, feats: torch.Tensor, return_attention: bool = False):
        """``(B,10,8)`` -> OKLab ``(B,3)`` (the bottleneck), optionally + attention."""
        b = feats.shape[0]
        tok = self.input_norm(self.input_proj(self._expand_angles(feats)))   # (B,10,d)
        obs = self.observer_token.expand(b, -1, -1)                  # (B,1,d)
        seq = torch.cat([tok, obs], dim=1)                          # (B,11,d) obs=idx 10
        attns = []
        for layer in self.layers:
            seq, attn = layer(seq, need_weights=return_attention)
            if return_attention:
                attns.append(attn)
        pooled = seq[:, -1] if self.cfg.pool == "observer" else seq.mean(dim=1)
        oklab = self._bound_oklab(self.to_bottleneck(pooled))       # (B,3)
        if return_attention:
            return oklab, torch.stack(attns, dim=1)                 # (B,L,H,11,11)
        return oklab

    # -- full autoencoder ----------------------------------------------------
    def forward(self, feats: torch.Tensor):
        """Return ``(recon (B,11,8), oklab (B,3))``."""
        oklab = self.encode(feats)
        recon = self.decoder(oklab).view(-1, N_TOKENS, self.cfg.raw_features)
        return recon, oklab

    @staticmethod
    def body_names() -> tuple[str, ...]:
        return BODY_NAMES


def build_model(cfg: ModelConfig) -> LocalSkyAutoencoder:
    return LocalSkyAutoencoder(cfg)
