"""Physics-weighted reconstruction loss + OKLab health metrics.

The loss is a Mean-Squared Error between the original and reconstructed Local Sky
Matrix, scaled per element by a physics weight tensor (mass x proximity x feature).
Wrapping angular columns (azimuth, ecliptic longitude of the bodies) use a
circular squared error ``2 - 2cos(delta)`` so there is no 0/360-degree
discontinuity; everything else is a plain squared error.
"""

from __future__ import annotations

import torch

from .features import circular_mask

_MASK = torch.from_numpy(circular_mask())          # (11,8) bool, CPU cache


def physics_weighted_mse(recon: torch.Tensor, target: torch.Tensor,
                         weight: torch.Tensor) -> torch.Tensor:
    """Weighted MSE with wrap-safe angular columns. All args ``(B,11,8)``."""
    mask = _MASK.to(recon.device)
    diff = recon - target
    plain = diff * diff
    circular = 2.0 - 2.0 * torch.cos(diff)          # ~diff^2 for small diff
    resid = torch.where(mask, circular, plain)
    return (weight * resid).mean()


@torch.no_grad()
def oklab_stats(oklab: torch.Tensor) -> dict[str, float]:
    """Health metrics to detect mode collapse to black/gray.

    ``mean_chroma`` (mean sqrt(a^2+b^2)) and ``std_L`` near zero => collapse.
    """
    L, a, b = oklab[:, 0], oklab[:, 1], oklab[:, 2]
    chroma = torch.sqrt(a * a + b * b)
    return {
        "mean_L": float(L.mean()),
        "std_L": float(L.std()),
        "mean_chroma": float(chroma.mean()),
        "mean_abs_a": float(a.abs().mean()),
        "mean_abs_b": float(b.abs().mean()),
    }
