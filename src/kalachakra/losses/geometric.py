"""
PyTorch composite geodesic loss (blueprint §5.1).

Differentiable counterpart of :mod:`kalachakra.losses.reference`. All three terms
operate on the local field E(t, s) with shape ``(batch, T, N, B, 5)`` (or any
leading batch dims) and are combined by fixed hyperparameter weights.

Requires PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

_TWO_PI = 2.0 * torch.pi


def _wrap(theta: torch.Tensor) -> torch.Tensor:
    return (theta + torch.pi) % _TWO_PI - torch.pi


def _split_field(field: torch.Tensor):
    direction = field[..., :3]
    offset_angle = torch.atan2(field[..., 4], field[..., 3])
    return direction, offset_angle


def geodesic_reconstruction_loss(recon: torch.Tensor, target: torch.Tensor,
                                 eps: float = 1e-7) -> torch.Tensor:
    """Clamped-arccos great-circle error on the horizon direction + offset angle."""
    r_dir, r_off = _split_field(recon)
    t_dir, t_off = _split_field(target)
    r_dir = F.normalize(r_dir, dim=-1, eps=eps)
    t_dir = F.normalize(t_dir, dim=-1, eps=eps)
    dot = (r_dir * t_dir).sum(-1).clamp(-1.0 + eps, 1.0 - eps)
    horizon = torch.arccos(dot)
    offset = _wrap(r_off - t_off).abs()
    return horizon.mean() + offset.mean()


def spectral_harmonic_loss(recon_seq: torch.Tensor, target_seq: torch.Tensor,
                           time_dim: int = 1) -> torch.Tensor:
    """Amplitude + phase divergence of the temporal spectrum (via rfft).

    FFT is unsupported in bf16/fp16 (and errors on MPS), so the transform runs in
    float32 with autocast disabled, then the result flows back to the caller.
    """
    with torch.autocast(device_type=recon_seq.device.type, enabled=False):
        r = torch.fft.rfft(recon_seq.float(), dim=time_dim)
        t = torch.fft.rfft(target_seq.float(), dim=time_dim)
        amp = (r.abs() - t.abs()).abs().mean()
        phase = _wrap(r.angle() - t.angle()).abs().mean()
    return amp + phase


def aspect_relational_invariance_loss(recon_lons: torch.Tensor,
                                      target_lons: torch.Tensor) -> torch.Tensor:
    """Rotation-invariant divergence of the pairwise angular-separation matrices."""
    def pairwise(lons: torch.Tensor) -> torch.Tensor:
        diff = lons.unsqueeze(-1) - lons.unsqueeze(-2)
        return _wrap(diff).abs()

    return (pairwise(recon_lons) - pairwise(target_lons)).abs().mean()


@dataclass
class LossWeights:
    geodesic: float = 1.0
    spectral: float = 0.5
    aspect: float = 0.5


class CompositeGeodesicLoss(torch.nn.Module):
    """Weighted sum of the three geometric loss terms; returns total + parts."""

    def __init__(self, weights: LossWeights | None = None, time_dim: int = 1):
        super().__init__()
        self.w = weights or LossWeights()
        self.time_dim = time_dim

    def forward(self, recon: torch.Tensor, target: torch.Tensor, *,
                recon_lons: torch.Tensor | None = None,
                target_lons: torch.Tensor | None = None):
        parts: dict[str, torch.Tensor] = {
            "geodesic": self.w.geodesic * geodesic_reconstruction_loss(recon, target),
            "spectral": self.w.spectral * spectral_harmonic_loss(
                recon, target, self.time_dim
            ),
        }
        if recon_lons is not None and target_lons is not None:
            parts["aspect"] = self.w.aspect * aspect_relational_invariance_loss(
                recon_lons, target_lons
            )
        total = sum(parts.values())
        parts["total"] = total
        return total, parts
