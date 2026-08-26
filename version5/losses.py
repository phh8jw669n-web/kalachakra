"""Unsupervised reconstruction loss + collapse telemetry.

The autoencoder is trained purely to reconstruct the local sky it just saw, so the
loss is a plain MSE between the decoded and the true ``(sin,cos)`` of each planet's
altitude and azimuth. An optional mass weighting lets the Sun and the giant planets
dominate the geometry the colours must preserve.

``oklab_stats`` (mean lightness, chroma, |a|, |b|) is **imported** from the root
autoencoder — it is range-agnostic and doubles as our mode-collapse alarm.
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
    """Per-body weight ``(12,)`` increasing with mass, bounded to roughly ``[0.5, 2.5]``
    (Sun & Jupiter heavy, Moon & Pluto light) — same scheme as train_v4. The two lunar
    nodes carry no mass, so they get the floor weight (0.5)."""
    lm = LOG_MASS_RAW / LOG_MASS_SCALE                          # 10 primaries
    lo, hi = float(lm.min()), float(lm.max())
    w = 0.5 + 2.0 * (lm - lo) / (hi - lo)
    w = np.concatenate([w, [0.5, 0.5]])                        # + Mean Node, True Node
    return torch.from_numpy(np.asarray(w, dtype=np.float32))


def reconstruction_loss(recon: torch.Tensor, target: torch.Tensor,
                        body_w: torch.Tensor | None = None) -> torch.Tensor:
    """Mass-weighted MSE over ``[B,12,4]`` reconstruction tensors."""
    resid = (recon - target) ** 2                              # [B,10,4]
    if body_w is not None:
        resid = resid * body_w.to(resid.device)[None, :, None]
    return resid.mean()
