"""The continuous stochastic generator — never a static dataset or grid.

Every training step draws a fresh batch of random floating-point ``(lat, lon, jd)``
triples (uniform over the configured ranges and time span) and runs them through the
topocentric ephemeris to build ``[B, 33]`` sky tensors on the fly. Because nothing is
ever cached or gridded, the SIREN cannot learn artificial spatial or temporal seams.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from . import ephemeris as ephem
from .config import DataConfig


class StochasticSky(IterableDataset):
    """Yields ``(sky [B,33], )`` — a fresh random batch of continuous skies per step."""

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
            sky = ephem.topocentric_tensor(lat, lon, jd)      # [B,33] float32
            yield (torch.from_numpy(sky),)


def build_dataloader(cfg: DataConfig, *, num_workers: int = 0) -> DataLoader:
    """A ``batch_size=None`` loader over :class:`StochasticSky` (dataset pre-batches)."""
    return DataLoader(StochasticSky(cfg), batch_size=None, num_workers=num_workers,
                      persistent_workers=num_workers > 0,
                      prefetch_factor=4 if num_workers > 0 else None)
