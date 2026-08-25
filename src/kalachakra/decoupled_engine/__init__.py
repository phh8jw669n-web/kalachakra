"""Decoupled projection engine.

A continuous, mesh-free replacement for the discrete 122,880-node / vector-quantised
architecture. Celestial geometry and terrestrial spatial mapping are decoupled into
two lightweight models:

* :class:`~kalachakra.decoupled_engine.sky_encoder.SkyEncoder` -- a Transformer that
  compresses the ten-body ephemeris state into one 512-D global tension vector.
* :class:`~kalachakra.decoupled_engine.earth_lens.EarthLensDecoder` -- a coordinate
  implicit field that maps (tension vector, lat, lon) to an OKLab colour, queryable
  at infinite resolution.

The pipeline streams continuous temporal slices straight from the Swiss-Ephemeris
loader; training is entirely self-supervised from wave-mechanics and spherical
differential geometry.
"""

from __future__ import annotations

from .bundle import build_models, load_checkpoint, save_checkpoint
from .config import (
    DataConfig,
    EarthLensConfig,
    EngineConfig,
    SkyEncoderConfig,
    TrainConfig,
)
from .earth_lens import EarthLensDecoder
from .sky_encoder import SkyEncoder

__all__ = [
    "EngineConfig", "SkyEncoderConfig", "EarthLensConfig", "DataConfig",
    "TrainConfig", "SkyEncoder", "EarthLensDecoder",
    "build_models", "save_checkpoint", "load_checkpoint",
]
