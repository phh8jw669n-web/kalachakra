"""Infinite continuous data generator for the Local Sky Autoencoder.

A PyTorch ``IterableDataset`` that yields endless random samples
``(timestamp_jd, lat, lon)`` and turns each into the physics tensors
``(features (10,8), target (11,8), weight (11,8))`` via :mod:`.features`.
Timestamps are uniform over the configured Julian-Day span; coordinates are
area-uniform over the sphere. Multi-worker safe (each worker gets its own RNG).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from .config import DataConfig
from .features import sample_sphere, sample_tensors


class LocalSkyDataset(IterableDataset):
    """Endless stream of Local Sky physics samples."""

    def __init__(self, cfg: DataConfig, epoch: int = 0):
        super().__init__()
        self.cfg = cfg
        self.epoch = int(epoch)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _rng(self) -> np.random.Generator:
        info = get_worker_info()
        wid = info.id if info else 0
        return np.random.default_rng(self.cfg.seed + 7919 * (self.epoch + 1) + wid)

    def __iter__(self):
        rng = self._rng()
        lo, hi = self.cfg.start_jd, self.cfg.end_jd
        while True:                                   # infinite stream
            jd = float(rng.uniform(lo, hi))
            lat, lon = sample_sphere(rng)
            feat, target, weight = sample_tensors(jd, lat, lon)
            yield (torch.from_numpy(feat),            # (10,8)
                   torch.from_numpy(target),          # (11,8)
                   torch.from_numpy(weight))          # (11,8)


def build_dataloader(cfg: DataConfig, batch_size: int, num_workers: int = 0,
                     epoch: int = 0) -> DataLoader:
    """DataLoader over the infinite stream (CPU tensors; the loop moves them)."""
    ds = LocalSkyDataset(cfg, epoch=epoch)
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                      drop_last=True, pin_memory=False)
