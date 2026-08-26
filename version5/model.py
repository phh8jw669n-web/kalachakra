"""The Sky-Energy Autoencoder (12-body "True Astrological Shape").

Encoder: each of the 12 bodies carries ``[alt, az, ecl_lon, ecl_lat, house_offset,
velocity]``; the five cyclic angles are expanded to ``(sin, cos)`` (seamless 0/2pi
wrap) and the velocity is kept as a scalar, then projected to ``d_model``. The 13th
``<OBSERVER>`` token is **data-driven**: a projection of the observer's high-frequency
geographic anchors ``[Ascendant, Midheaven, Vertex]`` (also as ``sin/cos``), so the
bottleneck can resolve city-level geography. A self-attention Transformer (block
imported from the root package) mixes the sequence; the observer token is funnelled
through a **3-neuron** OKLab head (``L`` sigmoid ``[0,1]``, ``a,b`` tanh ``[-1,1]``).

Decoder (training only): an MLP ``3 -> 64 -> 256 -> 12*4`` reconstructs the
``(sin,cos)`` of every body's altitude and azimuth. Loss falling proves the three
colours contain the entire local geometry.
"""

from __future__ import annotations

import torch
from torch import nn

from kalachakra.local_autoencoder.model import AttentionEncoderLayer  # reuse, don't rewrite

from .config import ModelConfig
from .sky_math import ANGULAR_COLS, SCALAR_COLS


class SkyEnergyEncoder(nn.Module):
    """``(features [B,12,6], observer [B,3]) -> OKLab [B,3]`` (the exported half)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.register_buffer("_ang", torch.tensor(ANGULAR_COLS, dtype=torch.long))
        self.register_buffer("_sca", torch.tensor(SCALAR_COLS, dtype=torch.long))
        body_in = 2 * len(ANGULAR_COLS) + len(SCALAR_COLS)       # 2*5 + 1 = 11
        self.input_proj = nn.Linear(body_in, d)
        self.input_norm = nn.LayerNorm(d)
        # data-driven <OBSERVER> token: sin/cos of Asc, MC, Vertex -> d_model
        self.observer_proj = nn.Linear(2 * cfg.obs_features, d)  # 2*3 = 6 -> d
        self.observer_norm = nn.LayerNorm(d)
        self.layers = nn.ModuleList([
            AttentionEncoderLayer(d, cfg.nhead, cfg.dim_feedforward, cfg.dropout)
            for _ in range(cfg.num_layers)
        ])
        self.to_bottleneck = nn.Linear(d, 3)

    def _expand_body(self, feats: torch.Tensor) -> torch.Tensor:
        """``[B,12,6]`` -> ``[B,12,11]``: cyclic angles -> ``(sin,cos)``, velocity
        ``tanh``-bounded.

        The five cyclic angles (altitude, azimuth, ecliptic lon/lat, house offset)
        become ``(sin,cos)`` pairs so there is no 359->0 deg wrap. The longitude
        velocity — already scaled by peak lunar speed (~15 deg/day) upstream — is
        squashed by ``tanh`` so it enters the Transformer strictly within
        ``[-1, 1]`` (baked into the ONNX graph, so the browser only supplies the
        scaled scalar)."""
        ang = feats.index_select(-1, self._ang)                  # [B,12,5]
        sca = feats.index_select(-1, self._sca)                  # [B,12,1] velocity/max_v
        return torch.cat([torch.sin(ang), torch.cos(ang), torch.tanh(sca)], dim=-1)

    @staticmethod
    def _expand_observer(obs: torch.Tensor) -> torch.Tensor:
        """``[B,3]`` angles (Asc,MC,Vx) -> ``[B,6]`` = ``[sin, cos]``."""
        return torch.cat([torch.sin(obs), torch.cos(obs)], dim=-1)

    @staticmethod
    def _bound_oklab(h: torch.Tensor) -> torch.Tensor:
        L = torch.sigmoid(h[..., :1])                            # [0,1]
        ab = torch.tanh(h[..., 1:])                              # [-1,1]
        return torch.cat([L, ab], dim=-1)

    def forward(self, features: torch.Tensor, observer: torch.Tensor,
                return_attention: bool = False):
        tok = self.input_norm(self.input_proj(self._expand_body(features)))   # [B,12,d]
        obs = self.observer_norm(
            self.observer_proj(self._expand_observer(observer))).unsqueeze(1)  # [B,1,d]
        seq = torch.cat([tok, obs], dim=1)                       # [B,13,d]  obs = idx 12
        attns = []
        for layer in self.layers:
            seq, attn = layer(seq, need_weights=return_attention)
            if return_attention:
                attns.append(attn)
        pooled = seq[:, -1] if self.cfg.pool == "observer" else seq.mean(dim=1)
        oklab = self._bound_oklab(self.to_bottleneck(pooled))    # [B,3]
        if return_attention:
            return oklab, torch.stack(attns, dim=1)              # [B,L,H,13,13]
        return oklab


class SkyEnergyAutoencoder(nn.Module):
    """Encoder + decoder. ``forward -> (recon [B,12,4], oklab [B,3])``."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = SkyEnergyEncoder(cfg)
        out = cfg.n_bodies * cfg.recon_features                  # 12*4 = 48
        dims = [3, *cfg.decoder_hidden, out]
        dec: list[nn.Module] = []
        for i in range(len(dims) - 1):
            dec.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                dec.append(nn.GELU())
        self.decoder = nn.Sequential(*dec)

    def encode(self, features, observer, return_attention: bool = False):
        return self.encoder(features, observer, return_attention=return_attention)

    def forward(self, features: torch.Tensor, observer: torch.Tensor):
        oklab = self.encoder(features, observer)
        recon = self.decoder(oklab).view(-1, self.cfg.n_bodies, self.cfg.recon_features)
        return recon, oklab


def build_model(cfg: ModelConfig) -> SkyEnergyAutoencoder:
    return SkyEnergyAutoencoder(cfg)
