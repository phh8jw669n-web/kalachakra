"""
Orthogonal spectral emission for the four temporal bands (blueprint Page 2).

The Micro / Fast / Cyclic / Macro band energies are mapped to four mathematically
orthonormal spectral basis functions over the visible range 380-750 nm. The
rendered spectrum is their additive optical superposition. Because the bases are
orthonormal on the sampling grid, the four scalar energies are perfectly
recoverable by projecting the spectrum back onto the bases — the bands never
destructively interfere and stay independently invertible.

Pure numpy.
"""

from __future__ import annotations

import numpy as np

BANDS = ("micro", "fast", "cyclic", "macro")
LAMBDA_MIN_NM = 380.0
LAMBDA_MAX_NM = 750.0


def wavelength_grid(n: int = 96) -> np.ndarray:
    """Uniform wavelength samples over the visible range (nm)."""
    return np.linspace(LAMBDA_MIN_NM, LAMBDA_MAX_NM, n)


def _orthonormal_basis(grid: np.ndarray, k: int = 4) -> np.ndarray:
    """``k`` orthonormal functions on ``grid`` (Gram-Schmidt on shifted Legendre).

    Orthonormal w.r.t. the discrete inner product ``sum(f*g)`` so recovery is an
    exact projection. Returns shape ``(k, len(grid))``.
    """
    x = np.linspace(-1.0, 1.0, grid.shape[0])
    raw = np.stack([np.polynomial.legendre.Legendre.basis(i)(x) for i in range(k)])
    # Modified Gram-Schmidt to orthonormalize on the discrete grid.
    basis = np.zeros_like(raw)
    for i in range(k):
        v = raw[i].copy()
        for j in range(i):
            v -= np.dot(v, basis[j]) * basis[j]
        basis[i] = v / np.linalg.norm(v)
    return basis


class SpectralTransducer:
    """Maps 4 band energies <-> a visible spectrum via an orthonormal basis."""

    def __init__(self, n_samples: int = 96):
        self.grid = wavelength_grid(n_samples)
        self.basis = _orthonormal_basis(self.grid, k=len(BANDS))  # (4, n)

    def emit(self, band_energies: dict[str, float] | np.ndarray) -> np.ndarray:
        """Additive superposition: spectrum = sum_b energy_b * basis_b."""
        e = self._as_vec(band_energies)
        return e @ self.basis                              # (n_samples,)

    def recover(self, spectrum: np.ndarray) -> dict[str, float]:
        """Recover the 4 band energies by projecting the spectrum onto the bases."""
        coeffs = self.basis @ np.asarray(spectrum, dtype=np.float64)
        return {b: float(coeffs[i]) for i, b in enumerate(BANDS)}

    @staticmethod
    def _as_vec(band_energies) -> np.ndarray:
        if isinstance(band_energies, dict):
            return np.array([band_energies[b] for b in BANDS], dtype=np.float64)
        v = np.asarray(band_energies, dtype=np.float64)
        if v.shape[-1] != len(BANDS):
            raise ValueError(f"expected {len(BANDS)} band energies")
        return v

    def gram(self) -> np.ndarray:
        """Gram matrix of the basis (should be the identity)."""
        return self.basis @ self.basis.T
