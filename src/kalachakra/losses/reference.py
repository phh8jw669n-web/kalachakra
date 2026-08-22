"""
Pure-numpy reference implementations of the composite geodesic loss (§5.1).

These define the *exact* mathematics of the three loss terms and are fully unit
tested. The PyTorch versions in :mod:`kalachakra.losses.geometric` mirror them
term for term; when in doubt this file is authoritative.

The local field E(t, s) packs, per body, a 5-vector
``[cosθcosh, sinθcosh, sinh, cosΔφ, sinΔφ]`` — a 3D horizon unit vector followed
by the Ascendant-offset angle on the unit circle.
"""

from __future__ import annotations

import numpy as np

from .. import geometry as geo


def _split_field(field: np.ndarray):
    """Split ``(..., 5)`` into (horizon unit vector ``(...,3)``, offset angle ``(...)``)."""
    direction = field[..., :3]
    offset_angle = np.arctan2(field[..., 4], field[..., 3])
    return direction, offset_angle


def geodesic_reconstruction_loss(recon: np.ndarray, target: np.ndarray) -> float:
    """Mean great-circle error between reconstructed and true local fields.

    Combines the geodesic distance on the horizon direction with the angular
    error on the Ascendant offset. Directions are renormalized before the clamped
    ``arccos`` so the reconstruction need not emit perfectly unit vectors.
    """
    r_dir, r_off = _split_field(recon)
    t_dir, t_off = _split_field(target)
    r_dir = r_dir / np.clip(np.linalg.norm(r_dir, axis=-1, keepdims=True), 1e-9, None)
    t_dir = t_dir / np.clip(np.linalg.norm(t_dir, axis=-1, keepdims=True), 1e-9, None)
    horizon = geo.geodesic_distance(r_dir, t_dir)             # radians
    offset = geo.angular_separation(r_off, t_off)             # radians
    return float(np.mean(horizon) + np.mean(offset))


def spectral_harmonic_loss(recon_seq: np.ndarray, target_seq: np.ndarray) -> float:
    """Frequency-domain divergence along the temporal axis (axis 0).

    Penalizes both amplitude and phase mismatch of the retained spectrum, forcing
    the model to preserve exact periodic frequencies rather than blurring them.
    """
    r = np.fft.rfft(recon_seq, axis=0)
    t = np.fft.rfft(target_seq, axis=0)
    amp = np.mean(np.abs(np.abs(r) - np.abs(t)))
    phase = np.mean(np.abs(geo.wrap_angle(np.angle(r) - np.angle(t))))
    return float(amp + phase)


def aspect_relational_invariance_loss(recon_lons: np.ndarray,
                                      target_lons: np.ndarray) -> float:
    """Divergence of the multi-body angular-separation matrices.

    ``*_lons`` are per-body ecliptic longitudes (radians), shape ``(..., B)``.
    Comparing the full pairwise separation matrix makes the term invariant to any
    global rotation of the configuration, so it scores structural aspects
    (conjunctions ~0, oppositions ~pi) rather than absolute positions.
    """
    r_mat = geo.pairwise_angular_matrix(recon_lons)
    t_mat = geo.pairwise_angular_matrix(target_lons)
    return float(np.mean(np.abs(r_mat - t_mat)))


def composite_loss(recon: np.ndarray, target: np.ndarray, *,
                   recon_lons: np.ndarray | None = None,
                   target_lons: np.ndarray | None = None,
                   w_geo: float = 1.0, w_spec: float = 0.5,
                   w_aspect: float = 0.5) -> dict[str, float]:
    """Weighted sum of the three terms; returns each component and the total."""
    parts = {
        "geodesic": w_geo * geodesic_reconstruction_loss(recon, target),
        "spectral": w_spec * spectral_harmonic_loss(recon, target),
    }
    if recon_lons is not None and target_lons is not None:
        parts["aspect"] = w_aspect * aspect_relational_invariance_loss(
            recon_lons, target_lons
        )
    parts["total"] = sum(parts.values())
    return parts
