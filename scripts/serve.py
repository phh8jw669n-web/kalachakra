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

from kalachakra import constants as C                              # noqa: E402
from kalachakra.grid.geodesic import fibonacci_sphere             # noqa: E402
from kalachakra.serving.api import create_app                     # noqa: E402
from kalachakra.serving.broadcast import BroadcastEngine          # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fields", type=Path, default=None,
                   help="npz with arrays 'potential' and 'shear' (N_nodes,)")
    p.add_argument("--nodes", type=int, default=C.N_SPATIAL_NODES)
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--demo", action="store_true", help="synthesize a field")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
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
        print("ERROR: provide --fields FILE or --demo", file=sys.stderr)
        return 2

    engine = BroadcastEngine(grid, potential, shear)

    try:
        import uvicorn
    except Exception:
        print("ERROR: uvicorn not installed. `pip install \"kalachakra[serve]\"`",
              file=sys.stderr)
        return 2

    app = create_app(engine, frame=args.frame)
    print(f"Serving Kalachakra broadcast on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
