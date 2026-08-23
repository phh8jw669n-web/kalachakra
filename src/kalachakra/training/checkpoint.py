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
from ..models.rvq import RVQConfig


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


# ---------------------------------------------------------------------------
# Quantized model (autoencoder + hierarchical residual VQ)
# ---------------------------------------------------------------------------

def save_quantized_model(path: str | Path, model, ae_cfg: AutoencoderConfig,
                         neighbors: np.ndarray, rvq_cfg: RVQConfig,
                         grid_xyz: np.ndarray | None = None,
                         extra: dict | None = None) -> Path:
    """Save a :class:`QuantizedSphericalAutoencoder` self-describingly.

    Persists both configs, the neighbor table, the grid, and the full state dict
    (encoder/decoder weights **and** the RVQ codebooks/EMA buffers) so the
    tokenizer reloads deterministically for offline inference.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": "kalachakra-qmodel-v1",
        "ae_config": asdict(ae_cfg),
        "rvq_config": asdict(rvq_cfg),
        "neighbors": np.asarray(neighbors, dtype=np.int64),
        "grid_xyz": None if grid_xyz is None else np.asarray(grid_xyz, np.float64),
        "state_dict": model.state_dict(),
        "extra": extra or {},
    }, path)
    return path


def load_quantized_model(path: str | Path, device: str | torch.device = "cpu"):
    """Rebuild a QuantizedSphericalAutoencoder. Returns (model, ae_cfg, rvq_cfg, grid_xyz)."""
    from ..models.quantized_autoencoder import QuantizedSphericalAutoencoder
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if ckpt.get("format") != "kalachakra-qmodel-v1":
        raise ValueError(f"not a kalachakra quantized checkpoint: {path}")
    ae_cfg = AutoencoderConfig(**ckpt["ae_config"])
    rvq_cfg = RVQConfig(**ckpt["rvq_config"])
    neighbors = np.asarray(ckpt["neighbors"])
    model = QuantizedSphericalAutoencoder(ae_cfg, neighbors, rvq_cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ae_cfg, rvq_cfg, ckpt.get("grid_xyz")
