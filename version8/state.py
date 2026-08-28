"""The 88-D state vector: 33-D local grounding + 55-D geometric chords.

* 33-D local: the 11 bodies' topocentric horizontal unit vectors (North, East, Zenith),
  from the self-contained analytic ephemeris.
* 55-D chords: the pairwise dot products v_i·v_j for all C(11,2)=55 unique body pairs —
  the mutual angular separations (conjunction=+1, opposition=-1) handed to the network
  explicitly so it need not learn cross-products internally.

The pair order is fixed (i<j, lexicographic) and MUST be identical in Python, JS and GLSL.
"""

from __future__ import annotations

import numpy as np

from .ephemeris import N_BODIES, topocentric_tensor

N_LOCAL = 33
N_CHORD = 55
STATE_DIM = 88

#: the 55 unique (i, j) body pairs, i < j — the canonical chord order.
PAIRS: list[tuple[int, int]] = [(i, j) for i in range(N_BODIES) for j in range(i + 1, N_BODIES)]
assert len(PAIRS) == N_CHORD


def chords_from_local(local: np.ndarray) -> np.ndarray:
    """``[N,33]`` topocentric vectors -> ``[N,55]`` pairwise dot products."""
    n = local.shape[0]
    v = local.reshape(n, N_BODIES, 3)
    out = np.empty((n, N_CHORD), dtype=local.dtype)
    for k, (i, j) in enumerate(PAIRS):
        out[:, k] = np.sum(v[:, i] * v[:, j], axis=1)
    return out


def topocentric_state(lat_deg, lon_deg, jd) -> np.ndarray:
    """Any ``(lat, lon, jd)`` -> the flat 88-D state ``[N,88]`` float32 (local ++ chords)."""
    local = topocentric_tensor(lat_deg, lon_deg, jd)      # [N,33] float32
    chords = chords_from_local(local)                     # [N,55]
    return np.concatenate([local, chords], axis=1).astype(np.float32)


def split_state(state):
    """Split an ``[N,88]`` state into its local ``[N,33]`` and chord ``[N,55]`` parts."""
    return state[..., :N_LOCAL], state[..., N_LOCAL:]
