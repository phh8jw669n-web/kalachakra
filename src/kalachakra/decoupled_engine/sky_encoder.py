"""The Sky Encoder: celestial geometry -> a 512-D global tension vector.

A small pre-norm Transformer encoder that treats the ten bodies as tokens and lets
multi-head self-attention compute all-to-all angular interference (conjunctions,
squares, trines, oppositions ...) directly from the continuous position/velocity
projections -- no hard-coded aspect table. A learnable per-body identity embedding
distinguishes the tokens, and a learnable global summary (CLS) token pools the
configuration into the tension vector via a residual MLP head.

The raw per-head attention matrices are inspectable (``return_attention=True`` or
:meth:`SkyEncoder.planetary_attribution`) so a downstream UI can read off how much
each planet drives the current global tension.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .config import SkyEncoderConfig
from .features import BODY_NAMES


class AspectAttentionLayer(nn.Module):
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
        attn_out, attn_w = self.attn(h, h, h, need_weights=need_weights,
                                     average_attn_weights=False)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x, attn_w                       # attn_w: (B, nhead, S, S) or None


class ResidualMLP(nn.Module):
    """Two-layer residual block used by the tension-vector head."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, x):
        return x + self.net(self.norm(x))


class SkyEncoder(nn.Module):
    """Ten-body celestial state ``(B, 10, 5)`` -> global tension vector ``(B, 512)``."""

    def __init__(self, cfg: SkyEncoderConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.embed = nn.Linear(cfg.in_features, d)
        self.embed_norm = nn.LayerNorm(d)
        # learnable identity of each body token + one global summary token
        self.body_embed = nn.Parameter(torch.zeros(1, cfg.n_bodies, d))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.body_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.layers = nn.ModuleList([
            AspectAttentionLayer(d, cfg.nhead, cfg.dim_feedforward, cfg.dropout)
            for _ in range(cfg.num_layers)
        ])
        self.head = nn.Sequential(
            ResidualMLP(d), nn.LayerNorm(d), nn.Linear(d, cfg.tension_dim),
        )

    def _tokens(self, celestial: torch.Tensor) -> torch.Tensor:
        """``(B, 10, 5)`` -> ``(B, 11, d)`` token sequence (CLS prepended)."""
        b = celestial.shape[0]
        tok = self.embed_norm(self.embed(celestial) + self.body_embed)  # (B,10,d)
        cls = self.cls_token.expand(b, -1, -1)                          # (B,1,d)
        return torch.cat([cls, tok], dim=1)                            # (B,11,d)

    def forward(self, celestial: torch.Tensor, return_attention: bool = False):
        """Return the ``(B, tension_dim)`` global tension vector.

        With ``return_attention`` also returns the stacked per-head attention
        ``(B, num_layers, nhead, 11, 11)`` (token 0 is the global summary; tokens
        1..10 are the bodies in :data:`~kalachakra.decoupled_engine.features.BODY_NAMES`
        order).
        """
        seq = self._tokens(celestial)
        attns = []
        use_ckpt = self.cfg.grad_checkpoint and self.training and not return_attention
        for layer in self.layers:
            if use_ckpt:
                seq, _ = checkpoint(layer, seq, use_reentrant=False)
            else:
                seq, attn_w = layer(seq, need_weights=return_attention)
                if return_attention:
                    attns.append(attn_w)
        summary = seq[:, 0]                                    # CLS -> global summary
        z = self.head(summary)
        if self.cfg.normalize_output:
            z = F.normalize(z, dim=-1)
        if return_attention:
            return z, torch.stack(attns, dim=1)               # (B,L,H,11,11)
        return z

    @torch.no_grad()
    def planetary_attribution(self, celestial: torch.Tensor) -> torch.Tensor:
        """Per-body attribution ``(B, 10)`` summing to 1 across the ten bodies.

        Reads the global summary token's attention onto each body, averaged over
        heads and layers -- a dynamic, table-free measure of which planets drive
        the current tension.
        """
        _z, attn = self.forward(celestial, return_attention=True)     # (B,L,H,11,11)
        cls_to_bodies = attn[:, :, :, 0, 1:]                          # (B,L,H,10)
        weights = cls_to_bodies.mean(dim=(1, 2))                      # (B,10)
        return weights / (weights.sum(dim=-1, keepdim=True) + 1e-9)

    @staticmethod
    def body_names() -> tuple[str, ...]:
        return BODY_NAMES


def build_sky_encoder(cfg: SkyEncoderConfig) -> SkyEncoder:
    return SkyEncoder(cfg)
