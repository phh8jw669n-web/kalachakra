"""OKLab -> sRGB conversion (Bjorn Ottosson's OKLab). Self-contained numpy.

The bottleneck lives in OKLab (``L in [0,1]``, ``a,b in [-0.5,0.5]``); this maps it
to displayable 8-bit sRGB for the inference API.
"""

from __future__ import annotations

import numpy as np

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
    lab = np.asarray(lab, dtype=np.float64)
    lms = (lab @ _LAB_TO_LMS.T) ** 3
    return lms @ _LMS_TO_RGB.T


def oklab_to_srgb(lab: np.ndarray) -> np.ndarray:
    lin = np.clip(oklab_to_linear_srgb(lab), 0.0, 1.0)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)


def oklab_to_srgb8(lab: np.ndarray) -> np.ndarray:
    return np.round(oklab_to_srgb(lab) * 255.0).astype(np.uint8)
