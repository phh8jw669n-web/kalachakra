"""Stochastic continuous generator — a fresh random batch of observer skies per step.

Each item is the 88-D observer-dependent target feature (33 local ++ 55 horizon-gated chords)
for a random ``(lat, lon, jd)``. The training loop feeds the local 33 (as 11x3 tokens) to the
model and uses the full 88-D for the isometric target distance.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from .config import DataConfig
from .state import target_features


class StochasticSky(IterableDataset):
    """Yields ``(feat [B,88],)`` from random uniform ``(lat, lon, jd)`` per step."""

    def __init__(self, cfg: DataConfig, gate_k: float):
        super().__init__()
        self.cfg = cfg
        self.gate_k = gate_k

    def _rng(self) -> np.random.Generator:
        info = get_worker_info()
        wid = info.id if info is not None else 0
        return np.random.default_rng([self.cfg.seed, wid])

    def __iter__(self):
        cfg = self.cfg
        rng = self._rng()
        b = cfg.batch
        while True:
            lat = rng.uniform(cfg.lat_min, cfg.lat_max, size=b)
            lon = rng.uniform(cfg.lon_min, cfg.lon_max, size=b)
            jd = rng.uniform(cfg.jd_start, cfg.jd_end, size=b)
            feat = target_features(lat, lon, jd, self.gate_k)     # [B,88] float32
            yield (torch.from_numpy(feat),)


def build_dataloader(cfg: DataConfig, gate_k: float, *, num_workers: int = 0) -> DataLoader:
    return DataLoader(StochasticSky(cfg, gate_k), batch_size=None, num_workers=num_workers,
                      persistent_workers=num_workers > 0,
                      prefetch_factor=4 if num_workers > 0 else None)
