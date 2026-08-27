"""The infinite Monte-Carlo dataset — on-the-fly, never touching disk.

Each iteration:

1. draws **one** random timestamp, quantised to the 24-second grid, from anywhere in
   the 10,256-year span (page 2: bounce between centuries so the net compresses the
   *geometry*, not a chronological sequence);
2. issues the **single** ten-call ephemeris query for that timestamp (page 4);
3. samples a batch of ``locations_per_step`` observers uniform over the sphere's
   area, and broadcasts the horizon math over them in one vectorised shot.

It therefore yields an already-assembled mini-batch and is consumed with
``DataLoader(batch_size=None)``. This reconciles page 2 ("random per step") with page
4 ("one query per batch of locations at a specific timestamp"): a *step* is one
Monte-Carlo timestamp, a *batch* is its cloud of observers. Workers march on
independent RNG streams, so N workers give N independent timestamps in flight.
"""

from __future__ import annotations

import functools

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from kalachakra.ephemeris import global_state as gs

from . import ephemeris as ephem
from . import sky_math
from .config import DataConfig


def _worker_init(worker_id: int, ephe_path: str | None, jpl_file: str | None) -> None:
    """Re-configure the ephemeris inside each worker.

    macOS/Windows spawn (do not fork) DataLoader workers, so the backend selection made
    in the parent process is not inherited — each worker must reconfigure ``pyswisseph``
    itself or it silently falls back to Moshier and clips the deep-time span.
    """
    if gs.ephemeris_available():
        ephem.configure(ephe_path=ephe_path, jpl_file=jpl_file)


class MonteCarloSky(IterableDataset):
    """Yields ``(state [B,50], jd)`` — one Monte-Carlo timestamp per step."""

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
        b = cfg.locations_per_step
        while True:
            jd = sky_math.random_jd_quantized(rng, cfg.start_jd, cfg.end_jd)
            ecl = ephem.ecliptic_state(jd)                  # the single query (12 bodies)
            eps = ephem.obliquity_rad(jd)
            gast = ephem.gast_radians(jd)
            lat, lon = sky_math.sample_locations(rng, b)
            state = sky_math.local_state(ecl, eps, gast, lat, lon)   # [B,50]
            yield (torch.from_numpy(state), float(jd))


def build_dataloader(cfg: DataConfig, *, num_workers: int = 0,
                     ephe_path: str | None = None, jpl_file: str | None = None,
                     pin_memory: bool = False) -> DataLoader:
    """A ``batch_size=None`` loader over :class:`MonteCarloSky` (dataset pre-batches)."""
    dataset = MonteCarloSky(cfg)
    kwargs: dict = dict(batch_size=None, num_workers=num_workers, pin_memory=pin_memory)
    if num_workers > 0:
        # functools.partial (not a lambda) so it survives pickling to spawned workers.
        kwargs["worker_init_fn"] = functools.partial(
            _worker_init, ephe_path=ephe_path, jpl_file=jpl_file)
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4
    return DataLoader(dataset, **kwargs)
