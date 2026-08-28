"""The balanced isometric loss for version8.

The 88-D state splits into a 33-D local part V and a 55-D chord part C. Their pairwise
distances are RMS-normalised so the 55 chords cannot dominate the 33 local vectors:

    d_local = ||V_A - V_B|| / sqrt(33)
    d_chord = ||C_A - C_B|| / sqrt(55)
    d_sky   = 0.5 * d_local + 0.5 * d_chord

The colour output must be isometric to d_sky: MSE(||Lab_A - Lab_B||, gamma * d_sky). A light
anchor keeps the mean colour near a pleasant neutral so the gamut is well-used. Distances are
taken over all pairs in the batch (torch.cdist) — the same objective as "two random skies
A, B" but with far more signal per step.
"""

from __future__ import annotations

import math

import torch

from .state import N_CHORD, N_LOCAL

__all__ = ["balanced_sky_distance", "isometric_loss", "anchor_loss", "color_stats"]

_INV_SQRT_LOCAL = 1.0 / math.sqrt(N_LOCAL)
_INV_SQRT_CHORD = 1.0 / math.sqrt(N_CHORD)
_ANCHOR = torch.tensor([55.0, 0.0, 0.0])           # pleasant neutral mid-grey L*a*b*


def balanced_sky_distance(state: torch.Tensor) -> torch.Tensor:
    """``[N,88]`` -> ``[N,N]`` balanced pairwise sky distance (0.5 local + 0.5 chord)."""
    v, c = state[:, :N_LOCAL], state[:, N_LOCAL:]
    d_local = torch.cdist(v, v) * _INV_SQRT_LOCAL
    d_chord = torch.cdist(c, c) * _INV_SQRT_CHORD
    return 0.5 * d_local + 0.5 * d_chord


def isometric_loss(state: torch.Tensor, color: torch.Tensor, gamma: float = 15.0) -> torch.Tensor:
    """MSE between the colour distance matrix and gamma * the balanced sky distance."""
    d_sky = balanced_sky_distance(state)
    d_lab = torch.cdist(color, color)
    return ((d_lab - gamma * d_sky) ** 2).mean()


def anchor_loss(color: torch.Tensor) -> torch.Tensor:
    return ((color.mean(dim=0) - _ANCHOR.to(color.device)) ** 2).mean()


def color_stats(color: torch.Tensor) -> dict:
    c = color.detach()
    return {"mean_L": float(c[:, 0].mean()), "std_L": float(c[:, 0].std()),
            "std_a": float(c[:, 1].std()), "std_b": float(c[:, 2].std())}
