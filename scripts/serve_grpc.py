#!/usr/bin/env python3
"""
Phase 4 — run the Kalachakra gRPC control plane (blueprint §7.2).

The strongly typed counterpart to scripts/serve_radar.py (REST/WebSocket): serves
the CosmicWeather service (Health / Inspect / Telemetry) over a Parquet token
index built by scripts/build_index.py.

Requires:  pip install "kalachakra[grpc,index]"   (grpcio + pyarrow duckdb h3)

Example:
    python scripts/build_index.py --out data/index --nodes 256 --frames 3000
    python scripts/serve_grpc.py --index data/index --port 50051
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
    p.add_argument("--port", type=int, default=50051)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    from kalachakra.serving.grpc_server import grpc_available, serve
    if not grpc_available():
        print('ERROR: grpcio not installed. `pip install "kalachakra[grpc]"`.',
              file=sys.stderr)
        return 2
    if not (args.index / "tier1").exists():
        print(f"WARNING: no tier1 data under {args.index}. Run build_index.py first.",
              file=sys.stderr)

    server = serve(str(args.index), host=args.host, port=args.port)
    print(f"Kalachakra gRPC CosmicWeather on {args.host}:{args.port}  (Ctrl-C to stop)")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(grace=1.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
