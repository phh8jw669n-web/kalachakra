#!/usr/bin/env python3
"""
Phase 3 analysis — load a TRAINED model and emit latent energy signatures.

Reloads a self-contained checkpoint written by train.py, encodes a real
spatio-temporal window into the 64-d latent manifold z(t, s), and derives the
geometric potential field (||z||) and temporal shear gradient (||dz/dt||) over
the Earth mesh for a real timestamp — the learned counterpart of the analytical
weather map. Optionally writes web/heatmap.json for the WebGL globe.

Requires:  pip install "kalachakra[train]"

Example:
    python scripts/analyze.py --checkpoint checkpoints/model_final.pt \\
        --date 2024-04-08T18:17 --out web/heatmap.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kalachakra.analysis import signatures                        # noqa: E402
from kalachakra.ephemeris import global_state, timeline           # noqa: E402
from kalachakra.ephemeris.calendar import format_jd, parse_datetime  # noqa: E402
from kalachakra.grid.geodesic import Grid                         # noqa: E402
from kalachakra.projection import spatial                         # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/model_final.pt"))
    p.add_argument("--date", default="now")
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--ephe-path", default=None)
    p.add_argument("--out", type=Path, default=None, help="write heatmap JSON here")
    p.add_argument("--top", type=int, default=5, help="report top-N singularities")
    p.add_argument("--sigma", type=float, default=2.5,
                   help="singularity threshold (median + sigma*MAD)")
    return p.parse_args(argv)


def _grid_from_xyz(xyz: np.ndarray) -> Grid:
    lat = np.arcsin(np.clip(xyz[:, 2], -1, 1))
    lon = np.arctan2(xyz[:, 1], xyz[:, 0])
    return Grid(xyz=xyz, lat=lat, lon=lon)


def main(argv=None) -> int:
    args = parse_args(argv)
    import torch
    from kalachakra.training.checkpoint import load_model

    if not args.checkpoint.exists():
        print(f"ERROR: no checkpoint at {args.checkpoint}. Run scripts/train.py first.",
              file=sys.stderr)
        return 2
    if not global_state.ephemeris_available():
        print("ERROR: pyswisseph not installed.", file=sys.stderr)
        return 2

    global_state.configure(mode="swiss" if args.ephe_path else "moshier",
                           ephe_path=args.ephe_path)

    model, cfg, grid_xyz = load_model(args.checkpoint)
    grid = _grid_from_xyz(np.asarray(grid_xyz))
    print(f"Loaded model: {cfg.n_nodes} nodes, latent={cfg.latent}, "
          f"{sum(p.numel() for p in model.parameters()):,} params")

    # Build a real temporal window centered on the requested instant.
    jd = parse_datetime(args.date)
    start = timeline.jd_to_frame(jd) - args.window // 2
    idx = np.arange(start, start + args.window)
    jds = timeline.frame_to_jd(idx)
    print(f"Encoding real window of {args.window} frames around {format_jd(jd)}...")

    fields = [spatial.project(global_state.global_state_frame(float(j)), float(j), grid)
              for j in jds]
    e = np.stack(fields, axis=0).reshape(args.window, grid.n_nodes, -1)  # (T,N,50)
    e_t = torch.from_numpy(e.astype(np.float32)).unsqueeze(0)            # (1,T,N,50)

    with torch.no_grad():
        z = model.encode(e_t)[0].cpu().numpy()      # (T, N, latent)

    sig = signatures.energy_signature(z, time_axis=0)  # potential/shear (T,N)
    mid = args.window // 2
    potential, shear = sig["potential"][mid], sig["shear"][mid]

    from kalachakra.analysis.anomaly import detect_singularities
    events = detect_singularities(sig["potential"], sig["shear"], sigma=args.sigma,
                                  max_events=args.top)
    print(f"\nLatent geometric potential (||z||) over {grid.n_nodes} nodes: "
          f"[{potential.min():.3f}, {potential.max():.3f}]")
    print(f"Latent temporal shear (||dz/dt||): max {shear.max():.3f}")
    print(f"\nTop {args.top} latent singularities in the window "
          f"(sigma={args.sigma}):")
    if not events:
        print("    (none above threshold — expected for a lightly-trained model; "
              "train longer / on more nodes, or lower --sigma)")
    for ev in events:
        y = format_jd(float(timeline.frame_to_jd(start + ev.time_index)))[:16]
        la = np.rad2deg(grid.lat[ev.node_index]); lo = np.rad2deg(grid.lon[ev.node_index])
        print(f"    {y}  node {ev.node_index:5d} ({la:+.1f},{lo:+.1f})  "
              f"score={ev.score:.2f}")

    if args.out:
        payload = {
            "jd": jd, "timestamp_utc": format_jd(jd),
            "lat_deg": np.rad2deg(grid.lat).round(4).tolist(),
            "lon_deg": np.rad2deg(grid.lon).round(4).tolist(),
            "potential": potential.round(5).tolist(),
            "shear": shear.round(5).tolist(),
            "cluster": [-1] * grid.n_nodes,
            "summary": {"source": "trained-latent", "nodes": grid.n_nodes},
        }
        args.out.write_text(json.dumps(payload))
        print(f"\nWrote {args.out} (from the trained latent field)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
