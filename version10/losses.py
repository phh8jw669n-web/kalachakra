"""The observer-dependent isometric loss for version10.

The colour field is trained so that the perceptual distance between two observers' colours
tracks a *fixed, observer-dependent* sky distance:

    d_local = ||V_A - V_B|| / sqrt(39)                 39-D topocentric local vectors (13 tokens)
    d_rel   = ||R_A - R_B|| / sqrt(78)                 78-D HORIZON-GATED chords  (C(13,2))
    d_sky   = w_local * d_local + w_rel * d_rel        defaults 0.5 / 0.5
    L       = MSE( ||ab_A - ab_B|| , gamma * d_sky ) + anchor + iso-pair (v10.1 anti-winding)

The colour is a pure 2-D OKLab chroma output (v10.1: raw Cartesian (a,b) on a disk of radius cmax,
no hue angle — so the optimiser cannot wind the hue) — no luminance. Euclidean distance on (a,b)
is the perceptual OKLab chroma distance, and (OKLab being perceptually uniform) the metric is
perceptual. gamma is in OKLab units (~60x smaller than the old CIELab default).

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

__all__ = ["balanced_sky_distance", "isometric_loss", "anchor_loss", "color_stats", "tv_loss",
           "pair_sky_distance", "isometric_pair_loss"]

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


def isometric_loss(feat: torch.Tensor, color: torch.Tensor, gamma: float = 0.35,
                   w_local: float = W_LOCAL, w_rel: float = W_REL) -> torch.Tensor:
    """MSE between the 2-D OKLab (a,b) chroma distance matrix and gamma * the sky distance.

    ``color`` is ``[N,2]`` OKLab (a,b) from the polar OKLCH head — no luminance, so distance is
    the perceptually-uniform OKLCH cylindrical distance and brightness is never optimised."""
    d_sky = balanced_sky_distance(feat, w_local, w_rel)
    d_ab = torch.cdist(color, color)
    return ((d_ab - gamma * d_sky) ** 2).mean()


def anchor_loss(color: torch.Tensor) -> torch.Tensor:
    return ((color.mean(dim=0) - _ANCHOR.to(color.device)) ** 2).mean()


def tv_loss(color_a: torch.Tensor, color_b: torch.Tensor) -> torch.Tensor:
    """Total-variation smoothness: mean squared OKLab (a,b) change between two geographically
    neighbouring observers at the same instant. Penalising this discourages high-frequency
    spatial noise, so sharpness must come from genuine token structure (esp. the fast ASC/MC),
    not hallucination. ``color_a``, ``color_b`` are ``[N,2]`` for matched neighbour points.

    NOTE: this is the blunt (pull-to-zero) prior; v10.1 training uses ``isometric_pair_loss``
    instead, which references the true sky metric and so never dulls a genuine gradient."""
    return ((color_a - color_b) ** 2).sum(dim=-1).mean()


def pair_sky_distance(feat_a: torch.Tensor, feat_b: torch.Tensor, w_local: float = W_LOCAL,
                      w_rel: float = W_REL) -> torch.Tensor:
    """Per-row (matched-pair) sky distance between two ``[N,117]`` target-feature tensors —
    the elementwise counterpart of ``balanced_sky_distance`` (which is all-pairs)."""
    va, ca = feat_a[:, :N_LOCAL], feat_a[:, N_LOCAL:]
    vb, cb = feat_b[:, :N_LOCAL], feat_b[:, N_LOCAL:]
    d_local = torch.linalg.vector_norm(va - vb, dim=-1) * _INV_SQRT_LOCAL
    d_rel = torch.linalg.vector_norm(ca - cb, dim=-1) * _INV_SQRT_CHORD
    return w_local * d_local + w_rel * d_rel


def isometric_pair_loss(color_a: torch.Tensor, color_b: torch.Tensor, feat_a: torch.Tensor,
                        feat_b: torch.Tensor, gamma: float = 0.35, w_local: float = W_LOCAL,
                        w_rel: float = W_REL) -> torch.Tensor:
    """v10.1 anti-winding term. The SAME isometric objective as ``isometric_loss`` but applied to
    matched *spatial-neighbour* pairs: the OKLab colour gap must equal ``gamma * d_sky(pair)`` —
    no more. Hue winding (a whole colour-wheel turn while d_sky barely moves) grossly overshoots
    that target and is removed; a genuine gradient already meets it and is left untouched. Because
    the reference is the true sky distance, this is faithful to the energy signature by
    construction (unlike a pull-to-zero smoothness prior)."""
    d_ab = torch.linalg.vector_norm(color_a - color_b, dim=-1)
    d_sky = pair_sky_distance(feat_a, feat_b, w_local, w_rel)
    return ((d_ab - gamma * d_sky) ** 2).mean()


def color_stats(color: torch.Tensor) -> dict:
    c = color.detach()
    return {"mean_a": float(c[:, 0].mean()), "mean_b": float(c[:, 1].mean()),
            "std_a": float(c[:, 0].std()), "std_b": float(c[:, 1].std())}
