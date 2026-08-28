"""The version10 geometric layer.

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
  That is exactly the "topocentric relational event" version10 is built to render.

The pair order is fixed (i<j, lexicographic) and MUST match Python, JS and GLSL.
"""

from __future__ import annotations

import numpy as np

from .ephemeris import N_BODIES, topocentric_tensor

N_LOCAL = N_BODIES * 3          # 39 (13 tokens x N,E,Zenith)
N_CHORD = N_BODIES * (N_BODIES - 1) // 2   # 78 = C(13,2)
STATE_DIM = N_LOCAL + N_CHORD   # 117 — local (39) ++ gated chords (78) — the loss-target length

#: the 78 unique (i, j) token pairs, i < j — the canonical chord order.
PAIRS: list[tuple[int, int]] = [(i, j) for i in range(N_BODIES) for j in range(i + 1, N_BODIES)]
assert len(PAIRS) == N_CHORD
_PAIR_I = np.array([i for i, _ in PAIRS])   # vectorised chord indexing (same order as PAIRS)
_PAIR_J = np.array([j for _, j in PAIRS])


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
    dots = np.einsum("nki,nki->nk", v[:, _PAIR_I], v[:, _PAIR_J])   # [n,78] v_i·v_j per pair
    return (g[:, _PAIR_I] * g[:, _PAIR_J] * dots).astype(local.dtype)


def target_features(lat_deg, lon_deg, jd, gate_k: float) -> np.ndarray:
    """The 117-D observer-dependent loss-target features ``[N,117]`` (local ++ gated chords)."""
    local = local_vectors(lat_deg, lon_deg, jd)
    chords = gated_chords(local, gate_k)
    return np.concatenate([local, chords], axis=1).astype(np.float32)


def split_target(feat):
    """Split ``[N,117]`` target features into local ``[N,39]`` and gated-chord ``[N,78]``."""
    return feat[..., :N_LOCAL], feat[..., N_LOCAL:]
