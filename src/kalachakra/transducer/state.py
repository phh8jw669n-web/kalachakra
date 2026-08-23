"""
The Isomorphic Mathematical Transducer (blueprint Page 1).

Binds mathematical type to physical optical type and guarantees **lossless
mathematical invertibility**: the optical state fully encodes the underlying
physics, and :meth:`IsomorphicTransducer.invert` reconstructs the inputs — most
importantly the exact 64-d latent — to machine precision.

  scalar magnitude (||z||)     -> radiant flux        (Naka-Rushton, boundless)
  scalar rarity (NLL)          -> colour temperature  (Planckian locus)
  4 temporal band energies     -> visible spectrum     (orthonormal bases)
  vector field (div / curl)    -> fluid kinematics     (Helmholtz-Hodge)
  64-d latent tensor           -> topography           (orthonormal spherical harmonics)

Each channel is an orthogonal / bijective transform, so nothing is destroyed by
clipping or band interference. Requires numpy (+ scipy for the topography channel).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import kinematics, photometric, topography
from .spectral import SpectralTransducer


@dataclass
class OpticalState:
    """The complete optical encoding of a localized celestial field."""

    topography: np.ndarray                    # height field on the SH quad grid
    spectrum: np.ndarray                      # visible-band spectrum
    flux: float                               # radiant flux (potential)
    temperature: float                        # effective colour temperature (rarity)
    chromaticity: np.ndarray                  # CIE xy on the Planckian locus (render)
    vector_field: tuple[np.ndarray, np.ndarray] | None = None


class IsomorphicTransducer:
    """Forward/inverse transducer with lossless invertibility."""

    def __init__(self, n_modes: int = 64, n_theta: int = 16, n_phi: int = 32,
                 n_spectral: int = 96, flux_k: float = 1.0):
        self.quad = topography.make_quadrature(n_modes, n_theta, n_phi)
        self.spectral = SpectralTransducer(n_spectral)
        self.flux_k = flux_k

    # -- forward ----------------------------------------------------------
    def transduce(self, latent: np.ndarray, band_energies, rarity: float,
                  potential: float, div_field: np.ndarray | None = None,
                  curl_field: np.ndarray | None = None) -> OpticalState:
        topo_field = topography.synthesize(latent, self.quad)
        spectrum = self.spectral.emit(band_energies)
        flux = float(photometric.naka_rushton(potential, k=self.flux_k))
        temperature = float(photometric.rarity_to_temperature(rarity))
        chroma = photometric.planckian_xy(temperature)
        vec = None
        if div_field is not None and curl_field is not None:
            vec = kinematics.field_from_sources(div_field, curl_field)
        return OpticalState(topo_field, spectrum, flux, temperature, chroma, vec)

    # -- inverse ----------------------------------------------------------
    def invert(self, state: OpticalState) -> dict:
        """Reconstruct the inputs from the optical state (lossless)."""
        out = {
            "latent": topography.analyze(state.topography, self.quad),
            "band_energies": self.spectral.recover(state.spectrum),
            "potential": float(photometric.naka_rushton_inv(state.flux, k=self.flux_k)),
            # rarity recovered exactly from the stored effective temperature
            "rarity": float(photometric.temperature_to_rarity(state.temperature)),
        }
        if state.vector_field is not None:
            out["divergence"] = kinematics.divergence(*state.vector_field)
            out["curl"] = kinematics.curl(*state.vector_field)
        return out
