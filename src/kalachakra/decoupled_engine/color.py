"""OKLab <-> sRGB colour conversion (Bjorn Ottosson's OKLab).

The Earth Lens emits perceptually-uniform OKLab so the losses operate in a space
where Euclidean distance approximates perceived colour difference. For display /
WebGL upload the field is converted OKLab -> linear sRGB -> gamma sRGB and quantised
to bytes here. Pure numpy so inference has no torch dependency at the pixel stage.
"""

from __future__ import annotations

import numpy as np

# OKLab -> LMS' -> linear sRGB matrices (Ottosson, 2020).
_LAB_TO_LMS = np.array([
    [1.0,  0.3963377774,  0.2158037573],
    [1.0, -0.1055613458, -0.0638541728],
    [1.0, -0.0894841775, -1.2914855480],
], dtype=np.float64)

_LMS_TO_RGB = np.array([
    [4.0767416621, -3.3077115913,  0.2309699292],
    [-1.2684380046,  2.6097574011, -0.3413193965],
    [-0.0041960863, -0.7034186147,  1.7076147010],
], dtype=np.float64)


def oklab_to_linear_srgb(lab: np.ndarray) -> np.ndarray:
    """``(..., 3)`` OKLab -> ``(..., 3)`` linear-light sRGB (may be out of gamut)."""
    lab = np.asarray(lab, dtype=np.float64)
    lms = lab @ _LAB_TO_LMS.T
    lms = lms ** 3
    return lms @ _LMS_TO_RGB.T


def _linear_to_gamma(lin: np.ndarray) -> np.ndarray:
    lin = np.clip(lin, 0.0, 1.0)
    return np.where(lin <= 0.0031308, 12.92 * lin,
                    1.055 * np.power(lin, 1.0 / 2.4) - 0.055)


def oklab_to_srgb(lab: np.ndarray) -> np.ndarray:
    """``(..., 3)`` OKLab -> ``(..., 3)`` gamma-encoded sRGB in ``[0, 1]``."""
    return _linear_to_gamma(oklab_to_linear_srgb(lab))


def oklab_to_srgb8(lab: np.ndarray) -> np.ndarray:
    """``(..., 3)`` OKLab -> ``uint8`` sRGB, ready for a texture byte buffer."""
    return np.round(oklab_to_srgb(lab) * 255.0).astype(np.uint8)
