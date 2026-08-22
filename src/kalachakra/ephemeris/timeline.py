"""
Temporal axis of the simulation (blueprint §2.1, §2.2).

The timeline is a closed, deterministic sequence of Julian Days beginning at the
Kali Yuga epoch and advancing by exactly one Vighatika (24 s) per frame. Nothing
here touches an ephemeris; it only enumerates *when* samples are taken.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from .. import constants as C

#: One Vighatika expressed as a fraction of a day (the JD step per frame).
JD_STEP: float = C.VIGHATIKA_SECONDS / C.SECONDS_PER_DAY


def frame_to_jd(frame_index: np.ndarray | int) -> np.ndarray:
    """Vectorized frame-index -> Julian Day (UT).

    ``jd = epoch + frame_index * JD_STEP``. Accepts scalars or arrays.
    """
    idx = np.asarray(frame_index, dtype=np.float64)
    return C.KALI_YUGA_EPOCH_JD + idx * JD_STEP


def jd_to_frame(jd: np.ndarray | float) -> np.ndarray:
    """Inverse of :func:`frame_to_jd`, rounded to the nearest whole frame."""
    jd = np.asarray(jd, dtype=np.float64)
    return np.rint((jd - C.KALI_YUGA_EPOCH_JD) / JD_STEP).astype(np.int64)


def iter_chunk_ranges(chunk_frames: int) -> Iterator[tuple[int, int]]:
    """Yield ``(start_frame, end_frame)`` half-open ranges covering the timeline.

    Used by the storage writer (§3.2) to serialize the full 13.4-billion-frame
    global-state matrix into contiguous, memory-mappable chunk files.
    """
    if chunk_frames <= 0:
        raise ValueError("chunk_frames must be positive")
    total = C.total_temporal_frames()
    start = 0
    while start < total:
        end = min(start + chunk_frames, total)
        yield start, end
        start = end


def summary() -> dict[str, float]:
    """Human-readable summary of the temporal axis for logging / sanity checks."""
    bounds = C.timeline_bounds()
    return {
        "start_jd": bounds.start_jd,
        "end_jd": bounds.end_jd,
        "span_days": bounds.span_days,
        "span_years": C.TIMELINE_YEARS,
        "jd_step": JD_STEP,
        "vighatika_seconds": C.VIGHATIKA_SECONDS,
        "n_frames": bounds.n_frames,
    }
