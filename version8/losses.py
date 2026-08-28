"""The balanced isometric loss for version8.

The 88-D state splits into a 33-D local part V and a 55-D chord part C. Their pairwise
distances are RMS-normalised, then combined with a *local-dominant* weighting:

    d_local = ||V_A - V_B|| / sqrt(33)
    d_chord = ||C_A - C_B|| / sqrt(55)
    d_sky   = w_local * d_local + w_chord * d_chord      # defaults 0.7 / 0.3

Why local-dominant, not 0.5/0.5? The 55 chords are pairwise dot products of unit vectors, so
they are *rotation-invariant* — identical for every observer on Earth at a fixed instant
(std across the whole globe ~3e-6). They therefore contribute **zero** spatial variation: all
of the globe's geographic colour structure lives in the 33 local vectors. A 0.5/0.5 split
halved that lone spatial signal, capping fixed-time globe contrast at ~8.6 dE (a near-flat
gradient). Weighting local up to 0.7 (and gamma up to ~32) restores a vivid globe (~24 dE)
while the chords still carry ~a third of the *temporal* signal, tinting the palette as the
sky animates over time and eras.

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

#: default local-dominant weights (see module docstring for why not 0.5/0.5)
W_LOCAL = 0.7
W_CHORD = 0.3


def balanced_sky_distance(state: torch.Tensor, w_local: float = W_LOCAL,
                          w_chord: float = W_CHORD) -> torch.Tensor:
    """``[N,88]`` -> ``[N,N]`` balanced pairwise sky distance (local-dominant by default).

    The chords are observer-independent, so the ``w_local`` term is the *only* source of
    spatial (across-globe) colour variation; keep it dominant for a vivid globe.
    """
    v, c = state[:, :N_LOCAL], state[:, N_LOCAL:]
    d_local = torch.cdist(v, v) * _INV_SQRT_LOCAL
    d_chord = torch.cdist(c, c) * _INV_SQRT_CHORD
    return w_local * d_local + w_chord * d_chord


def isometric_loss(state: torch.Tensor, color: torch.Tensor, gamma: float = 32.0,
                   w_local: float = W_LOCAL, w_chord: float = W_CHORD) -> torch.Tensor:
    """MSE between the colour distance matrix and gamma * the balanced sky distance."""
    d_sky = balanced_sky_distance(state, w_local, w_chord)
    d_lab = torch.cdist(color, color)
    return ((d_lab - gamma * d_sky) ** 2).mean()


def anchor_loss(color: torch.Tensor) -> torch.Tensor:
    return ((color.mean(dim=0) - _ANCHOR.to(color.device)) ** 2).mean()


def color_stats(color: torch.Tensor) -> dict:
    c = color.detach()
    return {"mean_L": float(c[:, 0].mean()), "std_L": float(c[:, 0].std()),
            "std_a": float(c[:, 1].std()), "std_b": float(c[:, 2].std())}
