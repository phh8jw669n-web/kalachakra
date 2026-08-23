"""
Tensor -> topography transduction via spherical harmonics (blueprint Page 4).

The 64-d latent tensor is treated as the amplitude coefficients of the first 64
real spherical-harmonic modes. Since 1 + 3 + 5 + ... + 15 = (7+1)^2 = 64, this is
exactly degrees l = 0..7, all orders m. The Earth mesh height at each coordinate
is the linear superposition of those weighted harmonics.

Because the real SH form an **orthonormal** basis, the transform is losslessly
invertible: SH analysis (quadrature projection) recovers all 64 coefficients from
the height field to numerical precision — satisfying the Page-4 invertibility
constraint ("any localized spatial deformation can be inverted back into its
precise constituent latent coefficients").

Requires scipy (``sph_harm_y``). Real SH are built from the complex ones with the
standard Condon-Shortley + sqrt(2) real convention.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # pragma: no cover - optional dependency
    from scipy.special import sph_harm_y

    _HAS_SCIPY = True
except Exception:  # noqa: BLE001
    sph_harm_y = None
    _HAS_SCIPY = False

from .. import constants as C


def scipy_available() -> bool:
    return _HAS_SCIPY


def sh_modes(n: int = C.LATENT_DIM) -> list[tuple[int, int]]:
    """First ``n`` (l, m) modes in order l=0,1,2,... and m=-l..l."""
    modes = []
    l = 0
    while len(modes) < n:
        for m in range(-l, l + 1):
            modes.append((l, m))
            if len(modes) == n:
                break
        l += 1
    return modes


def real_sph_harm(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Real orthonormal spherical harmonic Y_lm on colatitude/azimuth grids."""
    if not _HAS_SCIPY:
        raise RuntimeError("scipy is required for spherical-harmonic topography.")
    if m == 0:
        return sph_harm_y(l, 0, theta, phi).real
    if m > 0:
        y = sph_harm_y(l, m, theta, phi)
        return np.sqrt(2.0) * (-1.0) ** m * y.real
    y = sph_harm_y(l, -m, theta, phi)
    return np.sqrt(2.0) * (-1.0) ** m * y.imag


@dataclass
class SphereQuadrature:
    """Gauss-Legendre (colatitude) x uniform (azimuth) grid for exact SH analysis."""

    theta: np.ndarray      # (Ntheta, Nphi) colatitude in [0, pi]
    phi: np.ndarray        # (Ntheta, Nphi) azimuth in [0, 2pi)
    weights: np.ndarray    # (Ntheta, Nphi) quadrature weights summing to 4pi
    basis: np.ndarray      # (n_modes, Ntheta, Nphi) precomputed real SH

    @property
    def n_modes(self) -> int:
        return self.basis.shape[0]


def make_quadrature(n_modes: int = C.LATENT_DIM,
                    n_theta: int = 16, n_phi: int = 32) -> SphereQuadrature:
    """Build a quadrature grid that integrates the first ``n_modes`` SH exactly.

    Defaults (16 Gauss nodes, 32 azimuths) exactly resolve l=0..7 (64 modes).
    """
    if not _HAS_SCIPY:
        raise RuntimeError("scipy is required for spherical-harmonic topography.")
    x, w = np.polynomial.legendre.leggauss(n_theta)   # nodes/weights on [-1, 1]
    theta_1d = np.arccos(x)                            # colatitude
    phi_1d = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    theta, phi = np.meshgrid(theta_1d, phi_1d, indexing="ij")
    # weight = w_theta (for d cos theta) * (2pi / n_phi) (uniform azimuth)
    wt = np.repeat(w[:, None], n_phi, axis=1) * (2.0 * np.pi / n_phi)

    modes = sh_modes(n_modes)
    basis = np.stack([real_sph_harm(l, m, theta, phi) for l, m in modes], axis=0)
    return SphereQuadrature(theta=theta, phi=phi, weights=wt, basis=basis)


def synthesize(coeffs: np.ndarray, quad: SphereQuadrature) -> np.ndarray:
    """Height field on the quadrature grid = sum_i coeff_i * Y_i."""
    coeffs = np.asarray(coeffs, dtype=np.float64)
    return np.einsum("i,ijk->jk", coeffs, quad.basis)


def analyze(height: np.ndarray, quad: SphereQuadrature) -> np.ndarray:
    """Recover SH coefficients from a height field (quadrature projection)."""
    return np.einsum("jk,ijk,jk->i", height, quad.basis, quad.weights)
