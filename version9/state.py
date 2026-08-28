"""The version9 geometric layer.

Two distinct roles, kept deliberately separate:

* **Model input** — the 33-D local sky, i.e. the 11 bodies' topocentric unit vectors
  (North, East, Zenith), reshaped to ``[11, 3]`` tokens. This (and only this) is what the
  self-attention network sees; it learns the relations itself.

* **Loss target** — an *observer-dependent* sky distance built from those same vectors:
  the 33-D local part PLUS 55-D **horizon-gated chords**
  ``R_ij = g_i * g_j * (v_i . v_j)`` with ``g_b = sigmoid(gate_k * zenith_b)``.

  Plain chords ``v_i . v_j`` are rotation-invariant, hence identical for every observer at a
  fixed instant (they add zero spatial variation — this was version8's flat-globe bug). The
  gate ``g_b`` — high when body ``b`` is above the horizon, ~0 when below — breaks that
  invariance: a conjunction overhead contributes, the same conjunction underfoot does not.
  That is exactly the "topocentric relational event" version9 is built to render.

The pair order is fixed (i<j, lexicographic) and MUST match Python, JS and GLSL.
"""

from __future__ import annotations

import numpy as np

from .ephemeris import N_BODIES, topocentric_tensor

N_LOCAL = 33
N_CHORD = 55
STATE_DIM = 88          # local (33) ++ gated chords (55) — the loss-target feature length

#: the 55 unique (i, j) body pairs, i < j — the canonical chord order.
PAIRS: list[tuple[int, int]] = [(i, j) for i in range(N_BODIES) for j in range(i + 1, N_BODIES)]
assert len(PAIRS) == N_CHORD


def local_vectors(lat_deg, lon_deg, jd) -> np.ndarray:
    """Any ``(lat, lon, jd)`` -> the 33-D topocentric local sky ``[N,33]`` float32."""
    return topocentric_tensor(lat_deg, lon_deg, jd)


def body_tokens(local: np.ndarray) -> np.ndarray:
    """``[N,33]`` -> ``[N,11,3]`` (North,East,Zenith) tokens — the model input."""
    return local.reshape(local.shape[0], N_BODIES, 3)


def horizon_gate(local: np.ndarray, gate_k: float) -> np.ndarray:
    """``[N,33]`` -> ``[N,11]`` visibility gate g_b = sigmoid(gate_k * zenith_b)."""
    zen = local.reshape(local.shape[0], N_BODIES, 3)[:, :, 2]
    return 1.0 / (1.0 + np.exp(-gate_k * zen))


def gated_chords(local: np.ndarray, gate_k: float) -> np.ndarray:
    """``[N,33]`` -> ``[N,55]`` horizon-gated chords R_ij = g_i*g_j*(v_i.v_j)."""
    n = local.shape[0]
    v = local.reshape(n, N_BODIES, 3)
    g = horizon_gate(local, gate_k)
    out = np.empty((n, N_CHORD), dtype=local.dtype)
    for k, (i, j) in enumerate(PAIRS):
        out[:, k] = g[:, i] * g[:, j] * np.sum(v[:, i] * v[:, j], axis=1)
    return out


def target_features(lat_deg, lon_deg, jd, gate_k: float) -> np.ndarray:
    """The 88-D observer-dependent loss-target features ``[N,88]`` (local ++ gated chords)."""
    local = local_vectors(lat_deg, lon_deg, jd)
    chords = gated_chords(local, gate_k)
    return np.concatenate([local, chords], axis=1).astype(np.float32)


def split_target(feat):
    """Split ``[N,88]`` target features into local ``[N,33]`` and gated-chord ``[N,55]``."""
    return feat[..., :N_LOCAL], feat[..., N_LOCAL:]
