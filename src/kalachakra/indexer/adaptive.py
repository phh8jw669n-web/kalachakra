"""Adaptive time-stepping engine (PRD architectural overview + page 3).

The clock cruises in coarse (1-hour) steps, continuously computing the first
derivative of the planetary spatial tensor G(t). When that velocity exceeds a
threshold — a fast lunar transit, a rapidly forming square, a shear event — it
downshifts to 24-second micro-frames and emits *every* frame of the event until
the tension stabilizes, then returns to coarse cruise. This preserves every
micro-transit while skipping millions of redundant stable-period calculations.

The "planetary spatial tensor" is the stacked geocentric body direction matrix
(N_BODIES x 3) from G(t); its per-hour rate of change (Frobenius norm, expressed
per hour so the threshold is resolution-independent) is the trigger metric.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import constants as C


def spatial_tensor(jd: float) -> np.ndarray:
    """The (N_BODIES, 3) geocentric direction tensor at ``jd`` (unit vectors)."""
    from ..ephemeris import global_state
    g = global_state.global_state_frame(float(jd))
    return np.asarray(g[:, :3], dtype=np.float64)


def tensor_velocity(t0: np.ndarray, t1: np.ndarray, dt_days: float) -> float:
    """Per-hour Frobenius rate of change between two direction tensors."""
    if dt_days <= 0:
        return 0.0
    dt_hours = dt_days * 24.0
    return float(np.linalg.norm(t1 - t0) / dt_hours)


@dataclass
class Tick:
    jd: float
    resolution_s: float          # 24.0 (fine) or 3600.0 (coarse)
    velocity: float              # per-hour tensor velocity that produced this tick
    fine: bool                   # True inside a high-velocity micro-window


class AdaptiveClock:
    """Yields :class:`Tick`s from ``start_jd`` to ``end_jd`` with adaptive resolution.

    ``on_downshift(jd, velocity)`` / ``on_upshift(jd, velocity, n_fine)`` callbacks
    let the caller loudly log every anomaly window (PRD page 3).
    """

    def __init__(self, start_jd: float, end_jd: float, coarse_s: float, fine_s: float,
                 threshold: float, max_fine_run: int = 20_000,
                 on_downshift=None, on_upshift=None):
        self.start_jd = float(start_jd)
        self.end_jd = float(end_jd)
        self.coarse_days = coarse_s / C.SECONDS_PER_DAY
        self.fine_days = fine_s / C.SECONDS_PER_DAY
        self.coarse_s = coarse_s
        self.fine_s = fine_s
        self.threshold = float(threshold)
        self.max_fine_run = int(max_fine_run)
        self.on_downshift = on_downshift
        self.on_upshift = on_upshift
        self.n_fine = 0
        self.n_coarse = 0
        self.n_events = 0

    def __iter__(self):
        jd = self.start_jd
        prev_t = spatial_tensor(jd)
        prev_jd = jd
        # emit the very first frame at coarse resolution
        yield Tick(jd, self.coarse_s, 0.0, fine=False)
        self.n_coarse += 1

        while jd < self.end_jd:
            nxt = min(jd + self.coarse_days, self.end_jd)
            t_next = spatial_tensor(nxt)
            vel = tensor_velocity(prev_t, t_next, nxt - prev_jd)

            if vel <= self.threshold:
                # stable cruise: accept the coarse step
                jd = nxt
                prev_t, prev_jd = t_next, jd
                if jd < self.end_jd or jd >= self.end_jd:
                    yield Tick(jd, self.coarse_s, vel, fine=False)
                    self.n_coarse += 1
                continue

            # high velocity -> downshift and walk 24 s micro-frames to `nxt`
            self.n_events += 1
            if self.on_downshift:
                self.on_downshift(jd, vel)
            fine_count = 0
            fjd = jd
            ft_prev = prev_t
            fprev_jd = prev_jd
            while fjd < nxt and fine_count < self.max_fine_run:
                fjd = min(fjd + self.fine_days, nxt)
                ft = spatial_tensor(fjd)
                fvel = tensor_velocity(ft_prev, ft, fjd - fprev_jd)
                yield Tick(fjd, self.fine_s, fvel, fine=True)
                self.n_fine += 1
                fine_count += 1
                ft_prev, fprev_jd = ft, fjd
                # stabilized inside the window -> stop micro-stepping early
                if fvel <= self.threshold and fjd < nxt:
                    break
            if self.on_upshift:
                self.on_upshift(fjd, vel, fine_count)
            jd = fjd
            prev_t, prev_jd = ft_prev, fjd

    @property
    def stats(self) -> dict:
        return {"coarse_ticks": self.n_coarse, "fine_ticks": self.n_fine,
                "downshift_events": self.n_events}
