"""
Scalar -> thermodynamic / photometric optics (blueprint Page 2).

Two invertible scalar channels:

* **Geometric potential -> radiant flux** via the Naka-Rushton photoreceptor
  saturation curve. Monotone and bijective on [0, inf) -> [0, 1), so no magnitude
  is ever destroyed by clipping to pure white/black; the raw magnitude is exactly
  recoverable (infinite dynamic range).
* **Rarity -> effective colour temperature** along the Planckian locus. Rarity
  maps log-uniformly to temperature (common = cool/red, rare = hot/UV), and the
  Planckian chromaticity has a monotone inverse (correlated colour temperature),
  so the rarity is recoverable from the optical state.

Pure numpy. Blackbody spectral radiance (Planck's law) is provided for the shader.
"""

from __future__ import annotations

import numpy as np

# ---- Naka-Rushton flux (potential magnitude) ----------------------------

def naka_rushton(x: np.ndarray, k: float = 1.0, n: float = 1.0) -> np.ndarray:
    """Forward: raw magnitude ``x >= 0`` -> flux in [0, 1). Monotone bijection."""
    x = np.asarray(x, dtype=np.float64)
    xn = np.power(np.clip(x, 0.0, None), n)
    return xn / (xn + k ** n)


def naka_rushton_inv(flux: np.ndarray, k: float = 1.0, n: float = 1.0) -> np.ndarray:
    """Inverse: flux in [0, 1) -> raw magnitude (exact recovery)."""
    f = np.clip(np.asarray(flux, dtype=np.float64), 0.0, 1.0 - 1e-12)
    return k * np.power(f / (1.0 - f), 1.0 / n)


# ---- Planckian locus (rarity -> colour temperature) ---------------------

T_COMMON_K = 1200.0     # frequent configs: cool, deep-red / infrared
T_RARE_K = 40000.0      # epochal singularities: hot, ultraviolet


def rarity_to_temperature(rarity: np.ndarray) -> np.ndarray:
    """Map rarity in [0, 1] to colour temperature (K), log-uniform + monotone."""
    r = np.clip(np.asarray(rarity, dtype=np.float64), 0.0, 1.0)
    return T_COMMON_K * (T_RARE_K / T_COMMON_K) ** r


def temperature_to_rarity(temperature: np.ndarray) -> np.ndarray:
    """Inverse of :func:`rarity_to_temperature`."""
    t = np.asarray(temperature, dtype=np.float64)
    return np.log(t / T_COMMON_K) / np.log(T_RARE_K / T_COMMON_K)


def planckian_xy(temperature: np.ndarray) -> np.ndarray:
    """Planckian locus CIE 1931 chromaticity (x, y) for a temperature (K).

    Kim et al. cubic approximation, valid ~1667-25000 K; extrapolated beyond for a
    continuous curve. Returns array shape ``(..., 2)``.
    """
    T = np.clip(np.asarray(temperature, dtype=np.float64), 1000.0, 25000.0)
    invT = 1e3 / T
    x = np.where(
        T <= 4000.0,
        -0.2661239 * invT ** 3 - 0.2343589 * invT ** 2 + 0.8776956 * invT + 0.179910,
        -3.0258469 * invT ** 3 + 2.1070379 * invT ** 2 + 0.2226347 * invT + 0.240390,
    )
    y = np.where(
        T <= 2222.0,
        -1.1063814 * x ** 3 - 1.34811020 * x ** 2 + 2.18555832 * x - 0.20219683,
        np.where(
            T <= 4000.0,
            -0.9549476 * x ** 3 - 1.37418593 * x ** 2 + 2.09137015 * x - 0.16748867,
            3.0817580 * x ** 3 - 5.87338670 * x ** 2 + 3.75112997 * x - 0.37001483,
        ),
    )
    return np.stack([x, y], axis=-1)


def cct_from_xy(xy: np.ndarray) -> np.ndarray:
    """Correlated colour temperature from CIE xy (McCamy's approximation).

    The monotone inverse of the Planckian locus used to recover the rarity.
    """
    xy = np.asarray(xy, dtype=np.float64)
    x, y = xy[..., 0], xy[..., 1]
    nn = (x - 0.3320) / (0.1858 - y)
    return 449.0 * nn ** 3 + 3525.0 * nn ** 2 + 6823.3 * nn + 5520.33


def blackbody_radiance(wavelength_nm: np.ndarray, temperature: float) -> np.ndarray:
    """Planck's law spectral radiance (arbitrary units) — reference for the shader."""
    h = 6.62607015e-34
    c = 2.99792458e8
    kB = 1.380649e-23
    lam = np.asarray(wavelength_nm, dtype=np.float64) * 1e-9
    return (2 * h * c ** 2) / (lam ** 5 * (np.exp(h * c / (lam * kB * temperature)) - 1.0))
