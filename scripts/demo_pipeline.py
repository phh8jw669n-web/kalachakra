#!/usr/bin/env python3
"""
End-to-end smoke demo of the Kalachakra pipeline using only numpy.

This exercises every dependency-light stage on a small synthetic timeline so you
can see data flow through the whole system without PyTorch, pyswisseph or a
90-day training run:

    synthetic G(t)  ->  binary store (BF16 + delta)  ->  ring buffer
      ->  analytical spatial projection E(t, s)
      ->  (stand-in latent = the projected field)
      ->  energy signatures (potential / shear)
      ->  singularity detection
      ->  broadcast engine point query

Run:  python scripts/demo_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kalachakra.analysis import anomaly, signatures            # noqa: E402
from kalachakra.ephemeris import timeline                       # noqa: E402
from kalachakra.ephemeris.global_state import encode_body       # noqa: E402
from kalachakra.grid import geodesic                            # noqa: E402
from kalachakra.projection import spatial                       # noqa: E402
from kalachakra.serving.broadcast import BroadcastEngine        # noqa: E402
from kalachakra.storage.binary_store import EphemerisStore      # noqa: E402
from kalachakra.storage.ring_buffer import RingBuffer           # noqa: E402


def synthetic_timeline(n_frames: int, seed: int = 0) -> np.ndarray:
    """Smoothly drifting synthetic G(t): shape (n_frames, N_BODIES, 7)."""
    rng = np.random.default_rng(seed)
    n_bodies = 10
    lam0 = rng.uniform(-np.pi, np.pi, n_bodies)
    rate = rng.uniform(-0.05, 0.05, n_bodies)          # rad per frame
    frames = np.empty((n_frames, n_bodies, 7), dtype=np.float32)
    for k in range(n_frames):
        for i in range(n_bodies):
            lam = lam0[i] + rate[i] * k
            frames[k, i] = encode_body(lam, bet=0.02 * np.sin(0.01 * k),
                                       r=1.0 + i, lam_dot=rate[i],
                                       bet_dot=0.0, r_dot=0.0)
    return frames


def main() -> None:
    n_frames = 48
    n_nodes = 400
    print(f"[1] Grid: {n_nodes} observer nodes "
          f"(production target = 122,880)")
    grid = geodesic.fibonacci_sphere(n_nodes)

    print(f"[2] Generating {n_frames} synthetic G(t) frames "
          f"(Vighatika step = {timeline.JD_STEP * 86400:.0f}s)")
    frames = synthetic_timeline(n_frames)

    tmp = Path(__file__).resolve().parents[1] / "data" / "_demo_store"
    store = EphemerisStore(tmp)
    store.write_chunk(0, frames)
    print(f"[3] Serialized to BF16 + delta-encoded store at {tmp}")

    print("[4] Streaming via ring buffer + projecting to E(t, s)")
    fields = []
    with RingBuffer(store, [0], max_prefetch=2) as rb:
        for start_frame, chunk in rb:
            for k in range(chunk.shape[0]):
                jd = float(timeline.frame_to_jd(start_frame + k))
                fields.append(spatial.project(chunk[k], jd, grid))
    field_seq = np.stack(fields, axis=0)               # (T, N, B, 5)
    print(f"    E(t,s) shape = {field_seq.shape}")

    # Stand-in "latent": flatten body/feature axes. The trained encoder would
    # replace this with the 64-d code z(t, s).
    z = field_seq.reshape(n_frames, n_nodes, -1)
    print(f"[5] Latent stand-in shape = {z.shape} "
          f"(trained model -> (..., {64}))")

    sig = signatures.energy_signature(z, time_axis=0)
    print(f"[6] Potential field range = "
          f"[{sig['potential'].min():.3f}, {sig['potential'].max():.3f}]; "
          f"shear max = {sig['shear'].max():.4f}")

    events = anomaly.detect_singularities(sig["potential"], sig["shear"],
                                          sigma=3.0, max_events=5)
    print(f"[7] Singularities detected: {len(events)}")
    for e in events[:3]:
        print(f"    frame={e.time_index} node={e.node_index} score={e.score:.2f}")

    engine = BroadcastEngine(grid, sig["potential"][0], sig["shear"][0])
    reading = engine.query(lat_deg=48.85, lon_deg=2.35)  # Paris
    print(f"[8] Broadcast query @ Paris -> node {reading.node_index}, "
          f"potential={reading.potential_index:.3f}, "
          f"shear={reading.shear_velocity:.4f}")

    print("\nDemo complete — the full numpy pipeline ran end to end.")


if __name__ == "__main__":
    main()
