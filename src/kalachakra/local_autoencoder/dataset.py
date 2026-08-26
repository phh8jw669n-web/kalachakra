"""Infinite continuous data generator for the Local Sky Autoencoder.

A PyTorch ``IterableDataset`` that yields endless random samples
``(timestamp_jd, lat, lon)`` and turns each into the physics tensors
``(features (10,8), target (11,8), weight (11,8))`` via :mod:`.features`.
Timestamps are uniform over the configured Julian-Day span; coordinates are
area-uniform over the sphere. Multi-worker safe (each worker gets its own RNG).
"""

from __future__ import annotations

import functools

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from .config import DataConfig
from .features import sample_sphere, sample_tensors, sample_tensors_from_ecl


def _worker_init(worker_id: int, ephe_path: str | None, jpl_file: str | None) -> None:
    """Configure the ephemeris backend inside each DataLoader worker.

    Essential on macOS/Windows, where workers are **spawned** (fresh processes that
    do not inherit the main process's Swiss-file configuration). Without this a
    worker would silently fall back to Moshier and fail on deep-time (BCE) dates.
    """
    from ..ephemeris import global_state as gs
    if gs.ephemeris_available():
        gs.configure_from_args(ephe_path=ephe_path, jpl_file=jpl_file)


class LocalSkyDataset(IterableDataset):
    """Endless stream of Local Sky physics samples.

    With ``sky_cache`` set, the (expensive) global positions are read from a
    precomputed :class:`~kalachakra.local_autoencoder.sky_cache.SkyCache` memmap
    instead of ``calc_ut`` -- typically a ~10x speedup with the Swiss backend --
    while azimuth/altitude are still derived live (identical feature definition).
    """

    def __init__(self, cfg: DataConfig, epoch: int = 0,
                 sky_cache: str | None = None):
        super().__init__()
        self.cfg = cfg
        self.epoch = int(epoch)
        self.sky_cache_path = sky_cache

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _rng(self) -> np.random.Generator:
        info = get_worker_info()
        wid = info.id if info else 0
        return np.random.default_rng(self.cfg.seed + 7919 * (self.epoch + 1) + wid)

    def __iter__(self):
        rng = self._rng()
        if self.sky_cache_path is not None:
            yield from self._iter_cached(rng)
        else:
            yield from self._iter_live(rng)

    def _iter_live(self, rng):
        lo, hi = self.cfg.start_jd, self.cfg.end_jd
        while True:                                   # infinite stream
            jd = float(rng.uniform(lo, hi))
            lat, lon = sample_sphere(rng)
            yield self._pack(sample_tensors(jd, lat, lon))

    def _iter_cached(self, rng):
        from .sky_cache import SkyCache
        cache = SkyCache(self.sky_cache_path)         # memmap opened in this worker
        n = cache.n_frames
        while True:
            frame = int(rng.integers(0, n))
            jd = cache.jd_of(frame)
            lat, lon = sample_sphere(rng)
            ecl = cache.read(frame)
            yield self._pack(sample_tensors_from_ecl(jd, lat, lon, ecl))

    @staticmethod
    def _pack(triple):
        feat, target, weight = triple
        return (torch.from_numpy(feat), torch.from_numpy(target),
                torch.from_numpy(weight))


def build_dataloader(cfg: DataConfig, batch_size: int, num_workers: int = 0,
                     epoch: int = 0, ephe_path: str | None = None,
                     jpl_file: str | None = None, prefetch_factor: int = 4,
                     pin_memory: bool = False, sky_cache: str | None = None
                     ) -> DataLoader:
    """DataLoader over the infinite physics stream.

    The physics generation is CPU-bound (dominated by ``calc_ut``), so with
    ``num_workers > 0`` it runs in parallel worker processes and keeps the GPU fed.
    Each worker re-configures the ephemeris via :func:`_worker_init` (spawn-safe).
    A ``sky_cache`` directory removes ``calc_ut`` from the loop entirely (biggest win
    for the Swiss backend). Tensors stay on the CPU; the loop moves them.
    """
    ds = LocalSkyDataset(cfg, epoch=epoch, sky_cache=sky_cache)
    kwargs: dict = dict(batch_size=batch_size, num_workers=num_workers,
                        drop_last=True, pin_memory=pin_memory)
    if num_workers > 0:
        kwargs["worker_init_fn"] = functools.partial(
            _worker_init, ephe_path=ephe_path, jpl_file=jpl_file)
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(ds, **kwargs)
