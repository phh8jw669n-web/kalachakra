"""Compact global-sky cache for the ten Local-Sky bodies (Sun..Pluto).

Training samples random instants across a 10,000-year span, and the only expensive
step is ``calc_ut`` (the ephemeris query) -- ~90% of a sample with the Swiss ``.se1``
backend, because random deep-time access defeats its internal segment cache. This
module precomputes the *global* ecliptic state once, on a regular time grid, into a
memory-mapped array, so the training loop reads it (a page-cached RAM hit) instead
of calling ``calc_ut``. The fast, location-dependent horizontal coordinates
(azimuth/altitude) are still derived per sample from the cached positions via
``swe_azalt`` (a pure coordinate transform -- it never touches the ``.se1`` files),
so the training and inference feature definitions stay identical.

The cache stores only ``(n_frames, 10, 4)`` float32
``[lon_deg, lat_deg, lon_speed_deg_per_day, dist_au]`` -- a few GB, not the ~1.9 TB
of a full 24-second G(t) store (and, unlike that store, it contains the outer
planets Uranus/Neptune/Pluto that this engine needs).
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from .features import BODY_NAMES, ECL_COLS, N_BODIES

CACHE_FORMAT = "localsky-sky-cache-v1"
_N_COLS = 4


class SkyCache:
    """Read-only reader for a built sky cache (memory-mapped)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.meta = json.loads((self.path / "meta.json").read_text())
        self.start_jd = float(self.meta["start_jd"])
        self.cadence_days = float(self.meta["cadence_days"])
        self.n_frames = int(self.meta["n_frames"])
        self._mmap: np.memmap | None = None

    @property
    def mmap(self) -> np.memmap:
        # Opened lazily so it is created inside each DataLoader worker process.
        if self._mmap is None:
            self._mmap = np.memmap(self.path / "positions.dat", dtype=np.float32,
                                   mode="r", shape=(self.n_frames, N_BODIES, _N_COLS))
        return self._mmap

    def jd_of(self, frame: int) -> float:
        return self.start_jd + frame * self.cadence_days

    @property
    def end_jd(self) -> float:
        return self.jd_of(self.n_frames - 1)

    def read(self, frame: int) -> np.ndarray:
        """The ``(10, 4)`` ecliptic state at a frame."""
        return np.asarray(self.mmap[frame], dtype=np.float64)


# ---------------------------------------------------------------------------
# builder (parallel, sequential-per-worker so Swiss segment caching hits)
# ---------------------------------------------------------------------------
_W: dict = {}


def _init_worker(positions_path: str, shape: tuple, ephe_path, jpl_file) -> None:
    from ..ephemeris import global_state as gs
    if gs.ephemeris_available():
        gs.configure_from_args(ephe_path=ephe_path, jpl_file=jpl_file)
    _W["mm"] = np.memmap(positions_path, dtype=np.float32, mode="r+", shape=shape)


def _compute_chunk(args) -> tuple[int, int]:
    s, e, start_jd, cadence_days = args
    from .features import ecliptic_state
    mm = _W["mm"]
    for f in range(s, e):                           # contiguous -> Swiss cache hits
        mm[f] = ecliptic_state(start_jd + f * cadence_days).astype(np.float32)
    mm.flush()
    return s, e


def build_sky_cache(out_dir: str | Path, start_jd: float, end_jd: float,
                    cadence_hours: float = 1.0, ephe_path: str | None = None,
                    jpl_file: str | None = None, workers: int = 0,
                    chunk: int = 100_000, logger=print) -> Path:
    """Precompute the global ecliptic state on a time grid into a memmap cache."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cadence_days = cadence_hours / 24.0
    n_frames = int(math.floor((end_jd - start_jd) / cadence_days)) + 1
    shape = (n_frames, N_BODIES, _N_COLS)
    pos_path = out / "positions.dat"
    np.memmap(pos_path, dtype=np.float32, mode="w+", shape=shape).flush()   # allocate

    gb = n_frames * N_BODIES * _N_COLS * 4 / 1e9
    logger(f"building sky cache: {n_frames:,} frames @ {cadence_hours}h cadence "
           f"({gb:.2f} GB) over JD[{start_jd:.1f}..{end_jd:.1f}] -> {out}")
    chunks = [(s, min(s + chunk, n_frames), start_jd, cadence_days)
              for s in range(0, n_frames, chunk)]
    t0 = time.time()
    done = 0

    def _progress(s, e):
        nonlocal done
        done += e - s
        el = time.time() - t0
        rate = done / max(el, 1e-9)
        eta = (n_frames - done) / max(rate, 1e-9)
        logger(f"  {done:,}/{n_frames:,} frames ({100 * done / n_frames:.1f}%)  "
               f"{rate:,.0f} frames/s  elapsed {el / 60:.1f}m  ETA {eta / 60:.1f}m")

    if workers > 0:
        import multiprocessing as mp
        with mp.get_context().Pool(
                workers, initializer=_init_worker,
                initargs=(str(pos_path), shape, ephe_path, jpl_file)) as pool:
            for i, (s, e) in enumerate(pool.imap_unordered(_compute_chunk, chunks)):
                _progress(s, e)
    else:
        _init_worker(str(pos_path), shape, ephe_path, jpl_file)
        for c in chunks:
            s, e = _compute_chunk(c)
            _progress(s, e)

    meta = {
        "format": CACHE_FORMAT, "start_jd": start_jd, "end_jd": end_jd,
        "cadence_hours": cadence_hours, "cadence_days": cadence_days,
        "n_frames": n_frames, "n_bodies": N_BODIES, "cols": list(ECL_COLS),
        "bodies": list(BODY_NAMES),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    logger(f"sky cache done: {n_frames:,} frames in {(time.time() - t0) / 60:.1f}m")
    return out
