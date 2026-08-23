#!/usr/bin/env python3
"""
Phase 4 — run the Kalachakra radar service (blueprint §7).

Serves the FastAPI control plane (/health, /inspect) and the binary WebSocket
stream (/stream) over a Parquet token index built by scripts/build_index.py.
Open web/radar.html and point its WebSocket box at ws://HOST:PORT/stream.

Requires:  pip install "kalachakra[serve]" pyarrow duckdb h3

Example:
    python scripts/build_index.py --out data/index --nodes 256 --frames 3000
    python scripts/serve_radar.py --index data/index --port 8000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--index", type=Path, default=Path("data/index"),
                   help="Parquet token store root (from build_index.py)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    from kalachakra.serving.app import create_app, fastapi_available
    if not fastapi_available():
        print("ERROR: fastapi not installed. `pip install \"kalachakra[serve]\"`.",
              file=sys.stderr)
        return 2
    try:
        import uvicorn
    except Exception:
        print("ERROR: uvicorn not installed. `pip install \"kalachakra[serve]\"`.",
              file=sys.stderr)
        return 2

    if not (args.index / "tier1").exists():
        print(f"WARNING: no tier1 data under {args.index}. Run build_index.py first.",
              file=sys.stderr)

    app = create_app(str(args.index))
    print(f"Kalachakra radar on http://{args.host}:{args.port}  "
          f"(ws://{args.host}:{args.port}/stream)")
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
