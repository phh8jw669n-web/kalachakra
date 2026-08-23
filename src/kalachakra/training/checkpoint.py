"""
Self-contained model checkpoints (save / load the trained autoencoder).

A Trainer checkpoint stores optimizer/scheduler state for *resuming* a run. This
module stores everything needed to *reload the model for inference* with no other
context: the architecture config, the geodesic neighbor table, the grid, and the
weights. ``load_model`` rebuilds the network and returns it ready to encode.

Requires PyTorch.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ..models.autoencoder import AutoencoderConfig, SphericalAutoencoder


def save_model(path: str | Path, model: SphericalAutoencoder,
               cfg: AutoencoderConfig, neighbors: np.ndarray,
               grid_xyz: np.ndarray | None = None, extra: dict | None = None) -> Path:
    """Write a fully self-describing model checkpoint to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "kalachakra-model-v1",
        "config": asdict(cfg),
        "neighbors": np.asarray(neighbors, dtype=np.int64),
        "grid_xyz": None if grid_xyz is None else np.asarray(grid_xyz, dtype=np.float64),
        "state_dict": model.state_dict(),
        "extra": extra or {},
    }
    torch.save(payload, path)
    return path


def load_model(path: str | Path, device: str | torch.device = "cpu"):
    """Rebuild a :class:`SphericalAutoencoder` from a checkpoint.

    Returns ``(model, config, grid_xyz)`` with the model in eval mode on ``device``.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if ckpt.get("format") != "kalachakra-model-v1":
        raise ValueError(f"not a kalachakra model checkpoint: {path}")
    cfg = AutoencoderConfig(**ckpt["config"])
    neighbors = np.asarray(ckpt["neighbors"])
    model = SphericalAutoencoder(cfg, neighbors)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, cfg, ckpt.get("grid_xyz")
