"""Construction + checkpoint (de)serialisation for the paired decoupled models.

One checkpoint carries both models, the full :class:`EngineConfig`, and the
optimizer/scheduler/step so training is resumable and inference is a single load.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .config import EngineConfig
from .earth_lens import EarthLensDecoder
from .sky_encoder import SkyEncoder

CHECKPOINT_FORMAT = "kalachakra-decoupled-v1"


def build_models(cfg: EngineConfig):
    """Instantiate ``(SkyEncoder, EarthLensDecoder)`` from a config."""
    return SkyEncoder(cfg.sky), EarthLensDecoder(cfg.earth)


def save_checkpoint(path, sky: SkyEncoder, earth: EarthLensDecoder,
                    cfg: EngineConfig, step: int, *, optimizer=None,
                    scheduler=None, metrics: dict | None = None) -> Path:
    """Atomically write both models + training state to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "config": cfg.to_dict(),
        "sky_state": sky.state_dict(),
        "earth_state": earth.state_dict(),
        "step": int(step),
        "metrics": metrics or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    return path


def load_checkpoint(path, map_location="cpu"):
    """Load a checkpoint -> ``(sky, earth, cfg, payload)`` with weights restored."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    cfg = EngineConfig.from_dict(payload["config"])
    sky, earth = build_models(cfg)
    sky.load_state_dict(payload["sky_state"])
    earth.load_state_dict(payload["earth_state"])
    return sky, earth, cfg, payload
