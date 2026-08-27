"""Unsupervised reconstruction loss + collapse telemetry.

The autoencoder reconstructs the local sky it just saw. Each of the 12 celestial
bodies contributes an **equal** per-token MSE (over its ``(sin,cos)`` altitude &
azimuth) — no physical-mass weighting, so a body's influence comes purely from its
geometry and velocity, letting the Moon and the lunar nodes pull their weight. The
single ``<OBSERVER>`` token (Ascendant, Midheaven, Vertex) is upweighted by
``obs_weight`` so the 3-neuron bottleneck is forced to resolve local-horizon geography
instead of averaging continents into one wash:

    L = ( sum_i L_body_i  +  w_obs * L_observer ) / ( n_bodies + w_obs )

``oklab_stats`` is **imported** from the root autoencoder — range-agnostic, it doubles
as the mode-collapse alarm.
"""

from __future__ import annotations

import numpy as np
import torch

from kalachakra.local_autoencoder.features import (            # reuse mass constants
    LOG_MASS_RAW, LOG_MASS_SCALE,
)
from kalachakra.local_autoencoder.losses import oklab_stats    # reuse, don't rewrite

__all__ = ["mass_weights", "reconstruction_loss", "oklab_stats"]


def mass_weights() -> torch.Tensor:
    """Optional per-body mass weighting ``(12,)`` (Sun/Jupiter heavy, nodes light).

    Disabled by default (``--mass-w`` opt-in) — the rebalanced loss uses equal body
    weights so geometry, not mass, sets each body's importance. Kept for reproducibility.
    """
    lm = LOG_MASS_RAW / LOG_MASS_SCALE                          # 10 primaries
    lo, hi = float(lm.min()), float(lm.max())
    w = 0.5 + 2.0 * (lm - lo) / (hi - lo)
    w = np.concatenate([w, [0.5, 0.5]])                        # + Mean Node, True Node
    return torch.from_numpy(np.asarray(w, dtype=np.float32))


def reconstruction_loss(recon_body: torch.Tensor, target_body: torch.Tensor,
                        recon_obs: torch.Tensor, target_obs: torch.Tensor,
                        obs_weight: float = 3.0,
                        body_w: torch.Tensor | None = None) -> torch.Tensor:
    """Per-token MSE with an upweighted ``<OBSERVER>`` term.

    ``recon_body``/``target_body`` are ``[B,12,4]`` (``(sin,cos)`` alt & az per body);
    ``recon_obs``/``target_obs`` are ``[B,3,2]`` (``(sin,cos)`` of Asc/MC/Vertex). Each
    body contributes an equal per-token loss (mean over its 4 features); ``body_w``
    (default ``None`` = all 1.0) optionally restores mass weighting. The observer's
    per-token loss is scaled by ``obs_weight`` before the average, so it drives the
    gradients up front — this is what forces the bottleneck to resolve Asc/MC.
    """
    body_l = ((recon_body - target_body) ** 2).mean(dim=-1)    # [B,12] per-body loss
    if body_w is not None:
        bw = body_w.to(body_l.device)
        body_sum = (body_l * bw[None, :]).sum(dim=-1)          # [B]
        denom = float(bw.sum()) + obs_weight
    else:
        body_sum = body_l.sum(dim=-1)                          # [B]  (equal weights)
        denom = recon_body.shape[1] + obs_weight               # n_bodies + w_obs
    obs_l = ((recon_obs - target_obs) ** 2).mean(dim=(-1, -2))  # [B] observer loss
    total = (body_sum + obs_weight * obs_l) / denom            # [B]
    return total.mean()
