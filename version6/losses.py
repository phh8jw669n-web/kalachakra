"""The isometric mapping loss — the model as a perfect geometric mirror.

Within a batch of continuous skies the pairwise Euclidean distances in the 33-D physical
space must match the pairwise distances in the 3-D L*a*b* colour space (scaled by
``color_scale``). ``torch.cdist`` gives both distance matrices; the loss is their MSE.
The "Sky A / Sky B" framing of the PRD is the pairwise case — using the whole batch's
pairs is the same objective with far more signal per step.

A light **anchor** term fixes the translation gauge (an isometry is invariant to a rigid
shift of the colour cloud) so the output stays in the displayable L*a*b* range.
"""

from __future__ import annotations

import torch

__all__ = ["isometric_loss", "anchor_loss", "color_stats"]

_ANCHOR = torch.tensor([60.0, 0.0, 0.0])          # neutral mid-grey L*a*b*


def isometric_loss(sky: torch.Tensor, color: torch.Tensor,
                   color_scale: float = 20.0) -> torch.Tensor:
    """MSE between the physical (33-D) and colour (3-D) pairwise-distance matrices."""
    d_sky = torch.cdist(sky, sky)                 # [N,N]
    d_col = torch.cdist(color, color)             # [N,N]
    return ((d_col - color_scale * d_sky) ** 2).mean()


def anchor_loss(color: torch.Tensor) -> torch.Tensor:
    """Pull the batch's mean colour toward neutral L*=60 (gauge fix)."""
    return ((color.mean(dim=0) - _ANCHOR.to(color.device)) ** 2).mean()


def color_stats(color: torch.Tensor) -> dict:
    """Health metrics: mean L*, and the spread of a*/b* (collapse alarm)."""
    c = color.detach()
    return {
        "mean_L": float(c[:, 0].mean()),
        "std_L": float(c[:, 0].std()),
        "std_a": float(c[:, 1].std()),
        "std_b": float(c[:, 2].std()),
    }
