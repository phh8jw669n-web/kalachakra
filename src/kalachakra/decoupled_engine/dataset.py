"""Continuous temporal-slice streaming dataset for the decoupled engine.

An :class:`~torch.utils.data.IterableDataset` that never materialises an epoch of
history: each sample is a short run of ``temporal_len`` consecutive ephemeris
frames (for the temporal-continuity loss) drawn from a random point on the
10,256-year timeline, paired with a fresh batch of area-uniform terrestrial
coordinates. It streams straight from the Swiss-Ephemeris loader via
:mod:`.features`, so memory stays O(temporal_len x n_bodies + points_per_frame)
regardless of timeline length.

Each yielded sample is the triple::

    celestial  (temporal_len, 10, 5)   wrap-continuous body state
    jds        (temporal_len,)         Julian Days of the slice (float64)
    coords     (points_per_frame, 2)   random (lat, lon) radians

which the default collate batches to ``(B, T, 10, 5)``, ``(B, T)`` and
``(B, P, 2)``.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from .config import DataConfig
from .features import celestial_features_batch, sample_sphere_coords


class CelestialTerrestrialStream(IterableDataset):
    """Streams ``(celestial, jds, coords)`` slices over the continuous timeline."""

    def __init__(self, cfg: DataConfig, device: str | torch.device = "cpu",
                 epoch: int = 0):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(device)
        self.epoch = int(epoch)

    def set_epoch(self, epoch: int) -> None:
        """Reseed the stream for a new epoch (fresh random slices + coords)."""
        self.epoch = int(epoch)

    def _worker(self):
        info = get_worker_info()
        if info is None:
            return 0, 1, self.device
        # An accelerator context cannot be shared across a forked worker, so worker
        # processes always emit CPU tensors; the training loop moves them.
        return info.id, info.num_workers, torch.device("cpu")

    def __iter__(self):
        wid, nw, dev = self._worker()
        rng = np.random.default_rng(self.cfg.seed + 7919 * (self.epoch + 1) + wid)
        T = self.cfg.temporal_len
        stride = self.cfg.stride_days
        span = self.cfg.end_jd - self.cfg.start_jd - (T - 1) * stride
        span = max(span, 0.0)
        n_slices = len(range(wid, self.cfg.samples_per_epoch, nw))   # shard evenly
        for _ in range(n_slices):
            start = self.cfg.start_jd + rng.uniform(0.0, span)
            jds = start + np.arange(T, dtype=np.float64) * stride
            cel = celestial_features_batch(jds)                     # (T, 10, 5)
            coords = sample_sphere_coords(self.cfg.points_per_frame, rng)  # (P, 2)
            yield (
                torch.as_tensor(cel, dtype=torch.float32, device=dev),
                torch.as_tensor(jds, dtype=torch.float64, device=dev),
                torch.as_tensor(coords, dtype=torch.float32, device=dev),
            )


def move_batch(batch, device: str | torch.device):
    """Move a ``(celestial, jds, coords)`` batch onto the target device."""
    dev = torch.device(device)
    cel, jds, coords = batch
    return (cel.to(dev, non_blocking=True),
            jds.to(dev, non_blocking=True),
            coords.to(dev, non_blocking=True))


def build_dataloader(cfg: DataConfig, batch_size: int, num_workers: int = 0,
                     device: str | torch.device = "cpu",
                     epoch: int = 0) -> DataLoader:
    """Build the streaming :class:`~torch.utils.data.DataLoader`.

    With ``num_workers == 0`` the stream emits tensors already on ``device``; with
    workers it emits CPU tensors (fork-safe) that :func:`move_batch` relocates.
    """
    stream_device = device if num_workers == 0 else "cpu"
    stream = CelestialTerrestrialStream(cfg, device=stream_device, epoch=epoch)
    return DataLoader(stream, batch_size=batch_size, num_workers=num_workers,
                      drop_last=True)
