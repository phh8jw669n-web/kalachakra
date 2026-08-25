#!/usr/bin/env python3
"""
Decoupled Projection Engine - live global energy-signature dashboard server.

Serves web/decoupled.html plus the inference API (global energy texture, pinpoint
query with planetary attribution, latent similarity search) over a trained Sky
Encoder + Earth Lens checkpoint. With no --checkpoint it runs in demo mode with a
randomly-initialised model so the dashboard is explorable before training finishes.

Requires:  pip install "kalachakra[train,serve]"  (torch + fastapi/uvicorn +
           pyswisseph). Deep-time spans need the Swiss / DE441 files.

Example:
    python scripts/serve_decoupled.py --checkpoint checkpoints/decoupled_2024/model_final.pt
    python scripts/serve_decoupled.py            # demo mode (random weights)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=None,
                   help="trained checkpoint (.pt); omit for demo mode")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8100)
    p.add_argument("--device", default="")
    p.add_argument("--bank-size", type=int, default=64,
                   help="historical snapshots indexed for latent similarity search")
    p.add_argument("--ephe-path", default=None)
    p.add_argument("--jpl-file", default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    from kalachakra.decoupled_engine.api import serve
    return serve(args.checkpoint, host=args.host, port=args.port, device=args.device,
                 bank_size=args.bank_size, ephe_path=args.ephe_path,
                 jpl_file=args.jpl_file)


if __name__ == "__main__":
    raise SystemExit(main())
