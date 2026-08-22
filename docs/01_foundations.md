# 01 — Foundations: timeline, units, and the state tensors

This note fixes the vocabulary and the numbers the rest of the system is built
on. All figures live in `kalachakra.constants` and are re-derived there so they
can be checked.

## Timeline

| Quantity | Value | Source |
|---|---|---|
| Epoch | 3102-02-18 BCE 00:00 UTC = **JD 588465.5** | Kali Yuga anchor (§2.1) |
| Span | **10,256** years → 7154 CE | §2.1 |
| Step | 1 Vighatika = **24 s** = `24/86400` d | §2.2 |
| Frames | `10256 · 365.25 · 86400 / 24` ≈ **1.349 × 10¹⁰** | `constants.total_temporal_frames()` |

```python
from kalachakra.ephemeris import timeline
timeline.summary()          # start/end JD, step, frame count
timeline.frame_to_jd(1000)  # vectorized frame -> Julian Day
```

## Why native units

- **Time — the Vighatika.** At a 24 s step the eastern horizon turns
  `360° / 86164.0905 s · 24 s ≈ 0.10027°` per frame
  (`constants.HORIZON_ADVANCE_DEG_PER_FRAME`). Sampling on this natural cadence
  keeps the wave sampling aligned with Earth's rotation and avoids aliasing that
  a 1-second or 1-hour grid would introduce.
- **Space — angular separation.** The Earth is a geodesic mesh
  (`kalachakra.grid.geodesic`); "distance" between nodes is the great-circle
  angle, never a Euclidean chord or a map-projected metric.

## The two tensors

- **Global state `G(t) ∈ ℝ^{10×7}`** — the whole solar system at one instant, one
  row per entity (`kalachakra.ephemeris.global_state`).
- **Local field `E(t, s) ∈ ℝ^{N×10×5}`** — `G(t)` projected onto the `N =
  122,880` observers (`kalachakra.projection.spatial`).

The decoupling is the point: `G(t)` is computed once per frame; `E(t, s)` is a
pure analytical broadcast, so the expensive ephemeris query never runs per
observer.
