"""
PyTorch data ingestion pipeline (blueprint §3.3).

A custom ``IterableDataset`` for continuous temporal streams: it pulls decoded
G(t) chunks from the asynchronous ring buffer, slices them into fixed-length
temporal windows, projects each window onto the observer mesh (§3.1) and yields
device-ready local-field tensors E(t, s). Sliding the window inside a chunk and
letting the ring buffer stream the next chunk overlaps CPU parsing with GPU
compute, keeping the pipeline saturated.

Requires PyTorch. The projection math is the numpy reference engine, so the
stream matches the GPU kernel bit-for-bit within BF16 tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from ..ephemeris import timeline
from ..grid.geodesic import Grid
from ..projection import spatial
from ..storage.binary_store import EphemerisStore
from ..storage.ring_buffer import RingBuffer


@dataclass
class StreamConfig:
    window_frames: int = 64        # temporal length T of each sample
    window_stride: int = 32        # hop between successive windows
    node_subsample: int | None = None  # train on a random node subset if set
    max_prefetch: int = 3


class EphemerisStream(IterableDataset):
    """Iterable stream of ``(E, longitudes)`` windows over the timeline."""

    def __init__(self, store: EphemerisStore, grid: Grid, cfg: StreamConfig,
                 chunk_start_frames: list[int] | None = None):
        super().__init__()
        self.store = store
        self.grid = grid
        self.cfg = cfg
        if chunk_start_frames is None:
            chunk_start_frames = [c.start_frame for c in store.chunks()]
        self.chunk_start_frames = chunk_start_frames

    def _shard_for_worker(self) -> list[int]:
        """Split chunks across DataLoader workers so they never overlap."""
        info = get_worker_info()
        if info is None:
            return self.chunk_start_frames
        return self.chunk_start_frames[info.id :: info.num_workers]

    def _emit_windows(self, start_frame: int, chunk: np.ndarray):
        """Yield windows from one decoded chunk of shape ``(n, N_BODIES, 7)``."""
        n = chunk.shape[0]
        T = self.cfg.window_frames
        for w0 in range(0, n - T + 1, self.cfg.window_stride):
            g_window = chunk[w0 : w0 + T]                     # (T, B, 7)
            fields = []
            lons = []
            for k in range(T):
                jd = float(timeline.frame_to_jd(start_frame + w0 + k))
                field = spatial.project(g_window[k], jd, self.grid)  # (N, B, 5)
                lon, _lat = spatial.decode_ecliptic(g_window[k])
                fields.append(field)
                lons.append(lon)
            e = np.stack(fields, axis=0)                      # (T, N, B, 5)
            lon_seq = np.stack(lons, axis=0)                  # (T, B)

            if self.cfg.node_subsample:
                sel = np.random.choice(self.grid.n_nodes,
                                       self.cfg.node_subsample, replace=False)
                e = e[:, sel]

            # Flatten body*feature into the channel axis: (T, N, B*5).
            e = e.reshape(e.shape[0], e.shape[1], -1)
            yield (
                torch.from_numpy(e.astype(np.float32)),
                torch.from_numpy(lon_seq.astype(np.float32)),
            )

    def __iter__(self):
        starts = self._shard_for_worker()
        with RingBuffer(self.store, starts, max_prefetch=self.cfg.max_prefetch) as rb:
            for start_frame, chunk in rb:
                yield from self._emit_windows(start_frame, chunk)
