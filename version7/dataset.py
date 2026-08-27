"""The structured, high-density node sampler that replaces v6's pure random sampler.

Every training step draws a fresh batch whose observers are a mix of

* **curated metropolitan hubs** (:mod:`version7.cities`), with a little spatial jitter,
* a **regional lat/lon lattice** (the macro-scale grid), also jittered, and
* a **uniform** remainder so the continuous field stays valid *between* nodes,

each paired with a random Julian Date across the ~10,000-year timeline. The 33-D topocentric
tensors are built on the fly by the reused :func:`version6.ephemeris.topocentric_tensor`.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from version6 import ephemeris as ephem

from .cities import unique_cities
from .config import DataConfig


def regional_grid(step_deg: float) -> np.ndarray:
    """A regular lat/lon lattice -> ``[N, 2]`` of (lat, lon) node centres in degrees."""
    lats = np.arange(-90.0 + step_deg / 2, 90.0, step_deg)
    lons = np.arange(-180.0 + step_deg / 2, 180.0, step_deg)
    la, lo = np.meshgrid(lats, lons, indexing="ij")
    return np.stack([la.ravel(), lo.ravel()], axis=1)


def city_nodes() -> np.ndarray:
    """The curated hubs as a ``[N, 2]`` (lat, lon) array in degrees."""
    return np.array([[c[1], c[2]] for c in unique_cities()], dtype=float)


class StructuredNodes(IterableDataset):
    """Yields ``(sky [B,33], )`` — a fresh structured batch of continuous skies per step."""

    def __init__(self, cfg: DataConfig):
        super().__init__()
        self.cfg = cfg
        self.cities = city_nodes()
        self.grid = regional_grid(cfg.grid_step_deg)

    def _rng(self) -> np.random.Generator:
        info = get_worker_info()
        wid = info.id if info is not None else 0
        return np.random.default_rng([self.cfg.seed, wid])

    def __iter__(self):
        cfg = self.cfg
        rng = self._rng()
        b = cfg.batch
        n_city = int(round(b * cfg.city_frac))
        n_grid = int(round(b * cfg.grid_frac))
        n_uni = max(0, b - n_city - n_grid)
        while True:
            parts = []
            if n_city:
                idx = rng.integers(0, len(self.cities), n_city)
                pts = self.cities[idx] + rng.normal(0.0, cfg.jitter_deg, (n_city, 2))
                parts.append(pts)
            if n_grid:
                idx = rng.integers(0, len(self.grid), n_grid)
                pts = self.grid[idx] + rng.normal(0.0, cfg.jitter_deg, (n_grid, 2))
                parts.append(pts)
            if n_uni:
                lat = rng.uniform(-90.0, 90.0, n_uni)
                lon = rng.uniform(-180.0, 180.0, n_uni)
                parts.append(np.stack([lat, lon], axis=1))
            node = np.concatenate(parts, axis=0)
            lat = np.clip(node[:, 0], -89.999, 89.999)
            lon = ((node[:, 1] + 180.0) % 360.0) - 180.0
            jd = rng.uniform(cfg.jd_start, cfg.jd_end, b)
            sky = ephem.topocentric_tensor(lat, lon, jd)     # [B,33] float32
            yield (torch.from_numpy(sky),)


def build_dataloader(cfg: DataConfig, *, num_workers: int = 0) -> DataLoader:
    return DataLoader(StructuredNodes(cfg), batch_size=None, num_workers=num_workers,
                      persistent_workers=num_workers > 0,
                      prefetch_factor=4 if num_workers > 0 else None)
