#!/usr/bin/env python3
"""
End-to-end smoke run of the Kalachakra pipeline on REAL astronomical data.

Every stage uses real planetary positions (Swiss Ephemeris / Moshier backend,
no data files required):

    real G(t) via pyswisseph
      -> BF16 + delta memory-mapped store  ->  ring-buffer stream
      -> analytical spatial projection E(t, s)
      -> real cosmic-weather signatures (resonance / tension / potential)
      -> real per-node weather map
      -> real singularity scan (finds actual eclipses)

Run:  python scripts/demo_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kalachakra.analysis import weather                         # noqa: E402
from kalachakra.ephemeris import global_state, timeline         # noqa: E402
from kalachakra.ephemeris.calendar import format_jd, parse_datetime  # noqa: E402
from kalachakra.grid import geodesic                            # noqa: E402
from kalachakra.projection import spatial                       # noqa: E402
from kalachakra.storage.binary_store import EphemerisStore      # noqa: E402
from kalachakra.storage.ring_buffer import RingBuffer           # noqa: E402


def main() -> None:
    if not global_state.ephemeris_available():
        print("pyswisseph not installed — run `pip install pyswisseph`.")
        return

    anchor = "2024-04-08T18:17:00Z"   # real total solar eclipse
    jd0 = parse_datetime(anchor)
    print(f"[0] Anchor: {format_jd(jd0)}  (real ephemeris: "
          f"{'Swiss' if global_state._MODE == 'swiss' else 'Moshier'})")

    # 1) Real weather signature at the anchor instant.
    sig = weather.frame_signature(jd0)
    print(f"[1] Real signature: resonance={sig.resonance:.2f} "
          f"tension={sig.tension:.2f} potential(R)={sig.potential:.3f} "
          f"eclipse={sig.eclipse['is_eclipse']} "
          f"(sun-moon {sig.eclipse['sun_moon_sep_deg']:.2f} deg)")

    # 2) Generate a real G(t) block at the native 24 s Vighatika step, store it,
    #    and confirm the BF16 + delta round-trip.
    n_frames = 240
    start_frame = timeline.jd_to_frame(jd0)
    idx = np.arange(start_frame, start_frame + n_frames)
    jds = timeline.frame_to_jd(idx)
    frames = global_state.global_state_batch(jds)
    store = EphemerisStore(Path(__file__).resolve().parents[1] / "data" / "_demo_store")
    store.write_chunk(int(start_frame), frames.astype(np.float32))
    back = store.read_chunk(int(start_frame))
    err = float(np.abs(back - frames).max())
    print(f"[2] Stored {n_frames} real frames (BF16+delta); "
          f"round-trip max abs error = {err:.2e}")

    # 3) Stream via ring buffer and project one frame to a small mesh.
    grid = geodesic.fibonacci_sphere(400)
    with RingBuffer(store, [int(start_frame)], max_prefetch=2) as rb:
        for sframe, chunk in rb:
            field = spatial.project(chunk[0], float(timeline.frame_to_jd(sframe)), grid)
            break
    print(f"[3] Projected E(t,s) shape = {field.shape} "
          f"(direction sub-vectors unit-norm: "
          f"{np.allclose(np.linalg.norm(field[..., :3], axis=-1), 1.0, atol=1e-6)})")

    # 4) Real per-node weather map.
    wm = weather.weather_map(jd0, grid)
    print(f"[4] Weather map potential range = "
          f"[{wm['potential'].min():.2f}, {wm['potential'].max():.2f}], "
          f"shear max = {wm['shear'].max():.3f}")

    # 5) Real singularity scan across the eclipse month.
    scan = np.arange(parse_datetime("2024-04-01"), parse_datetime("2024-04-30"), 0.25)
    scores = [(weather.frame_signature(float(j)).eclipse["solar_proximity"], j)
              for j in scan]
    best = max(scores, key=lambda t: t[0])
    print(f"[5] Peak solar-eclipse proximity in April 2024: "
          f"{format_jd(best[1])[:16]} (prox={best[0]:.3f})")

    print("\nReal end-to-end pipeline ran successfully on live ephemeris data.")


if __name__ == "__main__":
    main()
