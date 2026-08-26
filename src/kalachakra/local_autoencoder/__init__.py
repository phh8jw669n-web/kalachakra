"""Local Sky Autoencoder (train_v4).

A purely physics-and-kinematics-driven autoencoder. For a sample ``(jd, lat, lon)``
it builds the Local Sky Matrix of the ten primary bodies (Sun..Pluto), compresses it
through a self-attention Transformer into a 3-channel **OKLab colour** bottleneck,
and reconstructs the matrix from that colour under a physics-weighted MSE. No
astrology, no grids, no heuristics -- the network discovers the geometric standing
waves purely by compressing localized physical kinematics into colour.
"""

from __future__ import annotations

from .config import DataConfig, LocalSkyConfig, ModelConfig, TrainConfig
from .inference import LocalSkyInference
from .model import LocalSkyAutoencoder, build_model
from .sky_cache import SkyCache, build_sky_cache

__all__ = [
    "LocalSkyConfig", "ModelConfig", "DataConfig", "TrainConfig",
    "LocalSkyAutoencoder", "build_model", "LocalSkyInference",
    "SkyCache", "build_sky_cache",
]
