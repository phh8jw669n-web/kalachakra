"""
Unsupervised energy-signature mapping from the latent manifold (blueprint §6.1).

Once trained, the decoder is severed and the analysis operates directly on the
latent codes z(t, s). Two foundational scalar fields are derived — both pure
numpy and fully tested:

* **Geometric Potential Field** — the L2 norm ``||z||`` at each (t, s). Spikes
  flag rare mass convergences (e.g. multi-body stelliums).
* **Temporal Shear Gradient** — the norm of the first temporal derivative
  ``||dz/dt||``. Spikes flag violent phase transitions (reversals, eclipses).
"""

from __future__ import annotations

import numpy as np


def geometric_potential_field(z: np.ndarray, axis: int = -1) -> np.ndarray:
    """L2 norm of the latent vector along ``axis`` (the latent dimension).

    Input ``z`` of shape ``(..., LATENT)`` -> output of shape ``(...)``.
    """
    return np.linalg.norm(np.asarray(z, dtype=np.float64), axis=axis)


def temporal_shear_gradient(z: np.ndarray, time_axis: int = 0,
                            dt: float = 1.0, latent_axis: int = -1) -> np.ndarray:
    """Norm of the first-order temporal derivative of the latent field.

    Uses a central difference in the interior and one-sided differences at the
    endpoints (``numpy.gradient``), then takes the latent-space norm. Output has
    the same shape as ``z`` minus the latent axis.
    """
    z = np.asarray(z, dtype=np.float64)
    dz = np.gradient(z, dt, axis=time_axis)
    return np.linalg.norm(dz, axis=latent_axis)


def energy_signature(z: np.ndarray, time_axis: int = 0,
                     dt: float = 1.0) -> dict[str, np.ndarray]:
    """Convenience bundle of both scalar fields for a latent tensor.

    ``z`` shape: ``(T, N, LATENT)``. Returns potential and shear fields shaped
    ``(T, N)``.
    """
    return {
        "potential": geometric_potential_field(z, axis=-1),
        "shear": temporal_shear_gradient(z, time_axis=time_axis, dt=dt),
    }
