#!/usr/bin/env python3
"""
Phase 4 — run the Cosmic Weather Broadcast API (blueprint §7).

Loads (or synthesizes) per-node potential and shear fields for a chosen frame
and serves them over the REST broadcast API. With ``--demo`` it fabricates a
field so you can exercise the endpoints without a trained model.

Requires:  pip install "kalachakra[serve]"   (fastapi + uvicorn)

Example:
    python scripts/serve.py --demo --nodes 4096 --port 8000
    # then: curl 'http://localhost:8000/potential?lat=48.85&lon=2.35'
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kalachakra.grid.geodesic import fibonacci_sphere             # noqa: E402
from kalachakra.serving.api import create_app                     # noqa: E402
from kalachakra.serving.broadcast import BroadcastEngine          # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--date", default="now",
                   help="ISO datetime or 'now' (UTC) for the real field (default)")
    p.add_argument("--fields", type=Path, default=None,
                   help="npz with arrays 'potential' and 'shear' (N_nodes,)")
    p.add_argument("--nodes", type=int, default=8000)
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--demo", action="store_true",
                   help="synthesize a field instead of computing real geometry")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--ephe-path", default=None,
                   help="Swiss .se1 directory, DE431 (full timeline)")
    p.add_argument("--jpl-file", default=None,
                   help="DE441 .bsp file for the JPL backend (full timeline)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    grid = fibonacci_sphere(args.nodes)

    if args.fields:
        data = np.load(args.fields)
        potential, shear = data["potential"], data["shear"]
    elif args.demo:
        rng = np.random.default_rng(0)
        potential = np.abs(rng.normal(1.0, 0.3, args.nodes))
        shear = np.abs(rng.normal(0.5, 0.2, args.nodes))
    else:
        # Default: compute the REAL weather field from real planetary geometry.
        from kalachakra.analysis import weather
        from kalachakra.ephemeris import global_state
        from kalachakra.ephemeris.calendar import format_jd, parse_datetime

        if not global_state.ephemeris_available():
            print("ERROR: pyswisseph not installed. Run `pip install pyswisseph`, "
                  "or pass --demo for a synthetic field.", file=sys.stderr)
            return 2
        # Honor a saved full-span config (setup_full_span) so far-past/future
        # dates work; falls back to Moshier otherwise.
        global_state.configure_from_args(ephe_path=args.ephe_path,
                                         jpl_file=args.jpl_file)
        jd = parse_datetime(args.date)
        print(f"Computing real weather field for {format_jd(jd)} "
              f"over {args.nodes:,} nodes...", file=sys.stderr)
        wm = weather.weather_map(jd, grid)
        potential, shear = wm["potential"], wm["shear"]

    engine = BroadcastEngine(grid, potential, shear)

    try:
        import uvicorn
    except Exception:
        print("ERROR: uvicorn not installed. `pip install \"kalachakra[serve]\"`",
              file=sys.stderr)
        return 2

    app = create_app(engine, frame=args.frame)
    from kalachakra.serving.webui import mount_web_ui
    web = mount_web_ui(app, Path(__file__).resolve().parents[1] / "web")
    print(f"Serving Kalachakra broadcast on http://{args.host}:{args.port}")
    if web is not None:
        print(f"Open the WebGL globe at  http://{args.host}:{args.port}/ui/index.html")
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
