"""The observer-dependent isometric loss for version9.

The colour field is trained so that the perceptual distance between two observers' colours
tracks a *fixed, observer-dependent* sky distance:

    d_local = ||V_A - V_B|| / sqrt(33)                 33-D topocentric local vectors
    d_rel   = ||R_A - R_B|| / sqrt(55)                 55-D HORIZON-GATED chords
    d_sky   = w_local * d_local + w_rel * d_rel        defaults 0.5 / 0.5
    L       = MSE( ||ab_A - ab_B|| , gamma * d_sky ) + anchor

The colour is pure 2-D chroma (a*, b*) — no luminance — so distance lives entirely in the
CIE a*b* plane and brightness is never optimised.

Both terms vary across the globe at a fixed instant (the gate makes the chords observer-
dependent — see state.py), so the globe is never a flat smear and relational events (a
conjunction climbing above your horizon) create real, localized colour structure.

Design note — we deliberately do **not** use the network's own attention matrix as the loss
target (as one might read the v9 PRD): that is circular and collapses (the optimiser can zero
the loss by making every attention map identical). The attention is the *architecture*; the
loss targets a fixed geometric quantity that realises the PRD's stated physics robustly.
Distances are taken over all pairs in the batch via ``torch.cdist``.
"""

from __future__ import annotations

import math

import torch

from .state import N_CHORD, N_LOCAL

__all__ = ["balanced_sky_distance", "isometric_loss", "anchor_loss", "color_stats"]

_INV_SQRT_LOCAL = 1.0 / math.sqrt(N_LOCAL)
_INV_SQRT_CHORD = 1.0 / math.sqrt(N_CHORD)
_ANCHOR = torch.tensor([0.0, 0.0])                 # neutral chroma centre (a*=b*=0); no L*

W_LOCAL = 0.5
W_REL = 0.5


def balanced_sky_distance(feat: torch.Tensor, w_local: float = W_LOCAL,
                          w_rel: float = W_REL) -> torch.Tensor:
    """``[N,88]`` target features (local ++ gated chords) -> ``[N,N]`` sky distance."""
    v, c = feat[:, :N_LOCAL], feat[:, N_LOCAL:]
    d_local = torch.cdist(v, v) * _INV_SQRT_LOCAL
    d_rel = torch.cdist(c, c) * _INV_SQRT_CHORD
    return w_local * d_local + w_rel * d_rel


def isometric_loss(feat: torch.Tensor, color: torch.Tensor, gamma: float = 32.0,
                   w_local: float = W_LOCAL, w_rel: float = W_REL) -> torch.Tensor:
    """MSE between the 2-D a*b* chroma distance matrix and gamma * the sky distance.

    ``color`` is ``[N,2]`` (a*, b*) — there is no luminance, so the distance is taken purely in
    the chromatic plane and brightness is never penalised or rewarded."""
    d_sky = balanced_sky_distance(feat, w_local, w_rel)
    d_ab = torch.cdist(color, color)
    return ((d_ab - gamma * d_sky) ** 2).mean()


def anchor_loss(color: torch.Tensor) -> torch.Tensor:
    return ((color.mean(dim=0) - _ANCHOR.to(color.device)) ** 2).mean()


def color_stats(color: torch.Tensor) -> dict:
    c = color.detach()
    return {"mean_a": float(c[:, 0].mean()), "mean_b": float(c[:, 1].mean()),
            "std_a": float(c[:, 0].std()), "std_b": float(c[:, 1].std())}
