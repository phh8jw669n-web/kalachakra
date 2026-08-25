"""The Earth Lens Decoder: (tension vector, lat/lon) -> perceptual colour.

A coordinate-based implicit neural representation (INR). The terrestrial
coordinate is lifted onto the unit sphere (removing the meridian seam) and passed
through random Fourier features to defeat the spectral bias of plain MLPs, so the
field can resolve sharp local gradients such as Ascendant transitions. The Fourier
embedding is concatenated with the global tension vector and pushed through a deep
residual MLP with Gaussian (or sine/SIREN) activations, ending in a three-channel
OKLab colour ``(L, a, b)``.

The forward pass is fully coordinate-agnostic: it evaluates any batch of points --
one pinpoint, a local bounding box, or a global equirectangular grid -- with no
architectural change and no fixed spatial discretisation.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .config import EarthLensConfig
from .features import latlon_to_unit_vector


class FourierFeatures(nn.Module):
    """Random Fourier features: ``x -> [sin(2*pi x B), cos(2*pi x B)]``."""

    def __init__(self, in_dim: int, num: int, scale: float, learnable: bool):
        super().__init__()
        b = torch.randn(in_dim, num) * scale
        if learnable:
            self.freqs = nn.Parameter(b)
        else:
            self.register_buffer("freqs", b)
        self.out_dim = 2 * num

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2.0 * math.pi * (x @ self.freqs)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class _Sine(nn.Module):
    def __init__(self, omega0: float = 1.0):
        super().__init__()
        self.omega0 = omega0

    def forward(self, x):
        return torch.sin(self.omega0 * x)


class _Gaussian(nn.Module):
    def __init__(self, sigma: float):
        super().__init__()
        self.inv2s2 = 1.0 / (2.0 * sigma * sigma)

    def forward(self, x):
        return torch.exp(-x * x * self.inv2s2)


def _make_activation(cfg: EarthLensConfig) -> nn.Module:
    if cfg.activation == "sine":
        return _Sine(cfg.sine_omega0)
    if cfg.activation == "gauss":
        return _Gaussian(cfg.gauss_sigma)
    raise ValueError(f"activation must be 'sine' or 'gauss', got {cfg.activation!r}")


class INRBlock(nn.Module):
    """Residual implicit block: ``x + W2 act(W1 act(x))``."""

    def __init__(self, dim: int, cfg: EarthLensConfig):
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)
        self.act = _make_activation(cfg)

    def forward(self, x):
        return x + self.lin2(self.act(self.lin1(self.act(x))))


class EarthLensDecoder(nn.Module):
    """``(tension (M,512), latlon (M,P,2)) -> OKLab colour (M,P,3)``."""

    def __init__(self, cfg: EarthLensConfig):
        super().__init__()
        self.cfg = cfg
        self.fourier = FourierFeatures(cfg.coord_dim, cfg.num_fourier,
                                       cfg.fourier_scale, cfg.learnable_fourier)
        in_dim = self.fourier.out_dim + cfg.tension_dim
        self.input_proj = nn.Linear(in_dim, cfg.hidden)
        self.act = _make_activation(cfg)
        self.blocks = nn.ModuleList([INRBlock(cfg.hidden, cfg)
                                     for _ in range(cfg.n_blocks)])
        self.head = nn.Linear(cfg.hidden, cfg.out_channels)
        if cfg.activation == "sine":
            self._siren_init()

    def _siren_init(self):
        """SIREN weight init so sine activations stay well-conditioned."""
        with torch.no_grad():
            fan = self.input_proj.in_features
            self.input_proj.weight.uniform_(-1.0 / fan, 1.0 / fan)
            for blk in self.blocks:
                for lin in (blk.lin1, blk.lin2):
                    bound = math.sqrt(6.0 / lin.in_features) / self.cfg.sine_omega0
                    lin.weight.uniform_(-bound, bound)

    def _broadcast(self, tension: torch.Tensor, latlon: torch.Tensor):
        if tension.dim() == 1:
            tension = tension.unsqueeze(0)
        if latlon.dim() == 2:
            latlon = latlon.unsqueeze(0)
        m = max(tension.shape[0], latlon.shape[0])
        if tension.shape[0] == 1 and m > 1:
            tension = tension.expand(m, -1)
        if latlon.shape[0] == 1 and m > 1:
            latlon = latlon.expand(m, -1, -1)
        return tension, latlon

    def forward(self, tension: torch.Tensor, latlon: torch.Tensor) -> torch.Tensor:
        """Evaluate the colour field. ``tension`` (M,512) or (512,); ``latlon``
        (M,P,2) or (P,2) in radians. Returns ``(M, P, 3)`` OKLab (or ``(P,3)`` when
        both inputs are unbatched)."""
        squeeze = tension.dim() == 1 and latlon.dim() == 2
        tension, latlon = self._broadcast(tension, latlon)
        m, p, _ = latlon.shape
        uv = latlon_to_unit_vector(latlon)                     # (M,P,3) continuous
        feats = self.fourier(uv)                               # (M,P,2*num)
        g = tension.unsqueeze(1).expand(m, p, tension.shape[-1])
        h = self.act(self.input_proj(torch.cat([feats, g], dim=-1)))
        for blk in self.blocks:
            h = blk(h)
        out = self.head(h)                                     # (M,P,3)
        if self.cfg.bound_output:
            L = 0.5 * (torch.tanh(out[..., :1]) + 1.0)         # [0,1]
            ab = self.cfg.ab_scale * torch.tanh(out[..., 1:])  # [-s,s]
            out = torch.cat([L, ab], dim=-1)
        return out.squeeze(0) if squeeze else out


def build_earth_lens(cfg: EarthLensConfig) -> EarthLensDecoder:
    return EarthLensDecoder(cfg)
