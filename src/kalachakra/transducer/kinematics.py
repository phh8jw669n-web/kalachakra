"""
Vector -> fluid kinematics via Helmholtz-Hodge (blueprint Page 3).

A 2D vector field on a periodic grid is split (FFT / Leray projection) into:
  * irrotational (curl-free) part  = grad phi   -> divergence carries the
    applying/separating signal (sink = converging = applying),
  * solenoidal (divergence-free) part = rot psi -> curl carries the orthogonal
    shear / vorticity,
  * harmonic (uniform) mean flow.

The decomposition is invertible: ``divergence`` of the field recovers the
applying/separating scalar and ``curl`` recovers the shear scalar, so the vector
channel losslessly stores both. Line Integral Convolution renders the field
densely (every pixel encodes the exact streamline orientation).

Pure numpy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _wavenumbers(shape):
    ny, nx = shape
    ky = 2 * np.pi * np.fft.fftfreq(ny)[:, None]
    kx = 2 * np.pi * np.fft.fftfreq(nx)[None, :]
    return kx, ky


def divergence(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Spectral divergence du/dx + dv/dy on a periodic grid."""
    kx, ky = _wavenumbers(u.shape)
    return np.fft.ifft2(1j * kx * np.fft.fft2(u) + 1j * ky * np.fft.fft2(v)).real


def curl(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Spectral scalar curl dv/dx - du/dy on a periodic grid."""
    kx, ky = _wavenumbers(u.shape)
    return np.fft.ifft2(1j * kx * np.fft.fft2(v) - 1j * ky * np.fft.fft2(u)).real


@dataclass
class HodgeParts:
    u_irrot: np.ndarray
    v_irrot: np.ndarray
    u_solen: np.ndarray
    v_solen: np.ndarray
    mean: tuple[float, float]

    def reconstruct(self):
        return (self.u_irrot + self.u_solen + self.mean[0],
                self.v_irrot + self.v_solen + self.mean[1])


def helmholtz_hodge(u: np.ndarray, v: np.ndarray) -> HodgeParts:
    """Decompose a periodic 2D vector field into its Hodge components."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    kx, ky = _wavenumbers(u.shape)
    k2 = kx ** 2 + ky ** 2
    k2[0, 0] = 1.0
    U, V = np.fft.fft2(u), np.fft.fft2(v)
    dot = kx * U + ky * V
    u_irr = np.fft.ifft2(kx * dot / k2).real
    v_irr = np.fft.ifft2(ky * dot / k2).real
    # irrotational part carries no mean (grad of a periodic potential); remove it
    # BEFORE forming the solenoidal residual so reconstruction is exact.
    u_irr -= u_irr.mean(); v_irr -= v_irr.mean()
    mean_u, mean_v = float(u.mean()), float(v.mean())
    u_sol = u - u_irr - mean_u
    v_sol = v - v_irr - mean_v
    return HodgeParts(u_irr, v_irr, u_sol, v_sol, (mean_u, mean_v))


def field_from_sources(div_field: np.ndarray, curl_field: np.ndarray):
    """Build a vector field whose divergence/curl equal the given scalar fields.

    Irrotational part solves nabla^2 phi = div; solenoidal part solves
    nabla^2 psi = -curl. The result is the inverse-recoverable vector channel:
    ``divergence`` / ``curl`` of the returned field recover the inputs exactly for
    Nyquist-free (odd-sized) grids. On even grids the (unrepresentable) Nyquist
    mode of the sources is dropped — supply odd grids or band-limited fields.
    """
    div_field = np.asarray(div_field, dtype=np.float64)
    curl_field = np.asarray(curl_field, dtype=np.float64)
    kx, ky = _wavenumbers(div_field.shape)
    k2 = kx ** 2 + ky ** 2
    k2[0, 0] = 1.0
    phi = -np.fft.fft2(div_field) / k2
    psi = np.fft.fft2(curl_field) / k2
    u = np.fft.ifft2(1j * kx * phi + 1j * ky * psi).real
    v = np.fft.ifft2(1j * ky * phi - 1j * kx * psi).real
    return u, v


def line_integral_convolution(u: np.ndarray, v: np.ndarray, noise: np.ndarray,
                              n_steps: int = 12, step: float = 0.5) -> np.ndarray:
    """Dense LIC of ``noise`` along the (u, v) streamlines (periodic wrap).

    Every output pixel averages the noise sampled along the local streamline, so
    the texture aligns with the field direction. Bilinear sampling, Euler steps.
    """
    ny, nx = noise.shape
    ys, xs = np.mgrid[0:ny, 0:nx].astype(np.float64)

    def sample(arr, x, y):
        x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int)
        fx = x - x0; fy = y - y0
        x0m, y0m = x0 % nx, y0 % ny
        x1m, y1m = (x0 + 1) % nx, (y0 + 1) % ny
        return (arr[y0m, x0m] * (1 - fx) * (1 - fy) + arr[y0m, x1m] * fx * (1 - fy)
                + arr[y1m, x0m] * (1 - fx) * fy + arr[y1m, x1m] * fx * fy)

    acc = noise.copy(); count = np.ones_like(noise)
    norm = np.hypot(u, v) + 1e-9
    un, vn = u / norm, v / norm
    for direction in (1.0, -1.0):
        x, y = xs.copy(), ys.copy()
        for _ in range(n_steps):
            du = sample(un, x, y); dv = sample(vn, x, y)
            x = (x + direction * step * du) % nx
            y = (y + direction * step * dv) % ny
            acc += sample(noise, x, y); count += 1.0
    return acc / count
