"""Progressive multi-scale temporal-resolution curriculum for base-model training.

Training strictly on 24-second frames from epoch 0 wastes billions of redundant
computations: the slow bodies barely move frame-to-frame, so the model spends its
first epochs re-learning a near-static field. This module implements *curriculum
learning* over the **temporal stride** instead — start coarse (a 24-hour step that
sweeps the whole 10,256-year timeline cheaply) and progressively refine toward the
24-second micro-resolution that captures high-velocity transit shears.

Two sampling regimes, chosen automatically by the per-epoch stride:

* **Continuous sweep** (stride >= 1 hour): walk the entire timeline start->end at
  the target stride, chopping the samples into training windows. One epoch == one
  full 10,256-year sweep.
* **Random micro-bursting** (stride < 1 hour): a full continuous sweep at these
  strides would be billions of frames, so instead sample ``n_windows`` random
  calendar windows of ``window_span_days`` each across the timeline and yield
  frames only within them at the target stride. This exposes the model to
  high-velocity geometry without an unbounded epoch.

The schedule (per the PRD):

    epoch  0-4   Solar       stride 24 h    continuous sweep
    epoch  5-9   Ascendant   stride  2 h    continuous sweep
    epoch 10-14  Navamsha    stride 24 min  micro-burst: 1000 x 6-month windows
    epoch 15-19  Degree      stride  4 min  micro-burst: 2000 x 1-month windows
    epoch 20+    Quantum     stride 24 s    micro-burst: 5000 x 1-week  windows

The dataset generates G(t) on the fly from the ephemeris (it does not read the
pre-built binary store, which only covers a fixed cadence) and projects each frame
onto the observer mesh with the same numpy reference engine as
:class:`~kalachakra.data.dataset.EphemerisStream`, so the two are interchangeable
from the trainer's point of view — both yield ``(E, longitudes)`` windows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from .. import constants as C
from ..ephemeris.global_state import global_state_batch
from ..grid.geodesic import Grid
from ..projection import spatial
from .dataset import StreamConfig

#: Strides at or above this switch to a continuous sweep; anything faster switches
#: to random micro-bursting (the "crucial micro-bursting constraint").
SUB_HOUR_THRESHOLD_S: float = 3600.0

_HOUR = 3600.0
_MINUTE = 60.0
#: Calendar spans used by the micro-burst phases (proleptic-Gregorian averages).
_SIX_MONTHS_DAYS = C.DAYS_PER_YEAR / 2.0        # ~182.6 days
_ONE_MONTH_DAYS = C.DAYS_PER_YEAR / 12.0        # ~30.4 days
_ONE_WEEK_DAYS = 7.0


@dataclass(frozen=True)
class CurriculumPhase:
    """One rung of the curriculum: the stride and how the timeline is sampled."""
    name: str
    stride_seconds: float
    mode: str                      # "continuous" | "microburst"
    n_windows: int = 0             # micro-burst only: random windows per epoch
    window_span_days: float = 0.0  # micro-burst only: calendar span of each window

    @property
    def human_stride(self) -> str:
        s = self.stride_seconds
        if s >= _HOUR:
            return f"{s / _HOUR:g} h"
        if s >= _MINUTE:
            return f"{s / _MINUTE:g} min"
        return f"{s:g} s"

    @property
    def human_plan(self) -> str:
        if self.mode == "continuous":
            return "continuous 10,256-year sweep"
        return (f"micro-burst: {self.n_windows} x {self.window_span_days:.4g}-day "
                f"random windows")


def curriculum_phase(epoch: int) -> CurriculumPhase:
    """Map a (0-based) training epoch to its temporal-resolution phase."""
    if epoch <= 4:
        return CurriculumPhase("solar", 24 * _HOUR, "continuous")
    if epoch <= 9:
        return CurriculumPhase("ascendant", 2 * _HOUR, "continuous")
    if epoch <= 14:
        return CurriculumPhase("navamsha", 24 * _MINUTE, "microburst",
                               n_windows=1000, window_span_days=_SIX_MONTHS_DAYS)
    if epoch <= 19:
        return CurriculumPhase("degree", 4 * _MINUTE, "microburst",
                               n_windows=2000, window_span_days=_ONE_MONTH_DAYS)
    return CurriculumPhase("quantum", float(C.VIGHATIKA_SECONDS), "microburst",
                           n_windows=5000, window_span_days=_ONE_WEEK_DAYS)


class CurriculumStream(IterableDataset):
    """Iterable ``(E, longitudes)`` windows whose temporal stride follows the epoch.

    The trainer calls :meth:`set_epoch` before iterating each epoch; the stride and
    sampling regime are then resolved from :func:`curriculum_phase`. Micro-burst
    anchors are drawn from a per-epoch-seeded RNG so a run is reproducible and the
    windows are sharded (never duplicated) across DataLoader workers.
    """

    def __init__(self, grid: Grid, cfg: StreamConfig,
                 start_jd: float | None = None, end_jd: float | None = None,
                 seed: int = 0, epoch: int = 0):
        super().__init__()
        self.grid = grid
        self.cfg = cfg
        b = C.timeline_bounds()
        self.start_jd = float(b.start_jd if start_jd is None else start_jd)
        self.end_jd = float(b.end_jd if end_jd is None else end_jd)
        self._seed = int(seed)
        self.epoch = int(epoch)

    def set_epoch(self, epoch: int) -> None:
        """Select the curriculum rung for the epoch about to be iterated."""
        self.epoch = int(epoch)

    # -- per-window projection (mirrors EphemerisStream._emit_windows) --------
    def _emit(self, jds: np.ndarray):
        g = global_state_batch(jds)                         # (T, B, 7)
        fields, lons = [], []
        for k in range(len(jds)):
            fields.append(spatial.project(g[k], float(jds[k]), self.grid))  # (N,B,5)
            lon, _lat = spatial.decode_ecliptic(g[k])
            lons.append(lon)
        e = np.stack(fields, axis=0)                        # (T, N, B, 5)
        lon_seq = np.stack(lons, axis=0)                    # (T, B)
        if self.cfg.node_subsample:
            sel = np.random.choice(self.grid.n_nodes,
                                   self.cfg.node_subsample, replace=False)
            e = e[:, sel]
        e = e.reshape(e.shape[0], e.shape[1], -1)           # (T, N, B*5)
        return (torch.from_numpy(e.astype(np.float32)),
                torch.from_numpy(lon_seq.astype(np.float32)))

    # -- window-start generators (yield the T Julian Days of each window) -----
    def _continuous_windows(self, phase: CurriculumPhase, wid: int, nw: int):
        stride = phase.stride_seconds / C.SECONDS_PER_DAY
        T, hop = self.cfg.window_frames, self.cfg.window_stride
        total = int(np.floor((self.end_jd - self.start_jd) / stride)) + 1
        i = w0 = 0
        while w0 + T <= total:
            if i % nw == wid:                               # shard across workers
                yield self.start_jd + (w0 + np.arange(T)) * stride
            i += 1
            w0 += hop

    def _microburst_windows(self, phase: CurriculumPhase, wid: int, nw: int):
        stride = phase.stride_seconds / C.SECONDS_PER_DAY
        span = phase.window_span_days
        T, hop = self.cfg.window_frames, self.cfg.window_stride
        per_burst = int(np.floor(span / stride)) + 1
        # per-epoch reproducible anchors; one global draw so worker shards agree.
        rng = np.random.default_rng(self._seed + 1009 * (self.epoch + 1))
        hi = max(self.start_jd, self.end_jd - span)
        anchors = rng.uniform(self.start_jd, hi, size=phase.n_windows)
        for bi in range(wid, phase.n_windows, nw):          # shard bursts by worker
            anchor = float(anchors[bi])
            w0 = 0
            while w0 + T <= per_burst:
                yield anchor + (w0 + np.arange(T)) * stride
                w0 += hop

    def __iter__(self):
        phase = curriculum_phase(self.epoch)
        info = get_worker_info()
        wid = info.id if info else 0
        nw = info.num_workers if info else 1
        gen = (self._continuous_windows(phase, wid, nw)
               if phase.mode == "continuous"
               else self._microburst_windows(phase, wid, nw))
        for jds in gen:
            yield self._emit(jds)
