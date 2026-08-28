"""Stochastic continuous generator — a fresh random batch of 88-D skies per step."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from .config import DataConfig
from .state import topocentric_state


class StochasticSky(IterableDataset):
    """Yields ``(state [B,88],)`` from random uniform ``(lat, lon, jd)`` per step."""

    def __init__(self, cfg: DataConfig):
        super().__init__()
        self.cfg = cfg

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
            state = topocentric_state(lat, lon, jd)          # [B,88] float32
            yield (torch.from_numpy(state),)


def build_dataloader(cfg: DataConfig, *, num_workers: int = 0) -> DataLoader:
    return DataLoader(StochasticSky(cfg), batch_size=None, num_workers=num_workers,
                      persistent_workers=num_workers > 0,
                      prefetch_factor=4 if num_workers > 0 else None)
