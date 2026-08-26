#!/usr/bin/env python3
"""
Build the global-sky cache for the Local Sky Autoencoder (train_v4).

Precomputes the geocentric ecliptic state of the ten bodies (Sun..Pluto) on a
regular time grid into a memory-mapped array, so train_v4 reads it instead of
calling the (expensive) Swiss ``calc_ut`` per sample -- typically a ~10x training
speedup. Only the *global* positions are cached (a few GB); the location-dependent
azimuth/altitude are still derived per sample at train time, so the feature
definition is identical to the live path.

Generation is parallelised and marches through time contiguously per worker, so the
Swiss segment cache stays hot (unlike random-access training).

Requires the Swiss ``.se1`` files (via --ephe-path) for the full BCE->CE span.

Example (full 10,256-year span, hourly, on a 12-core machine):
    python scripts/build_sky_cache.py \
        --start "-3101-02-18T00:00:00" --end "7155-02-18T00:00:00" \
        --cadence-hours 1 --workers 12 --ephe-path /path/to/ephe \
        --out data/sky_cache_1h
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="output cache directory")
    p.add_argument("--start", required=True, help="UTC ISO start (BCE '-YYYY-...' ok)")
    p.add_argument("--end", required=True, help="UTC ISO end")
    p.add_argument("--cadence-hours", type=float, default=1.0,
                   help="grid spacing (1.0 = hourly; coarser builds faster/smaller)")
    p.add_argument("--workers", type=int, default=0,
                   help="parallel generation workers (use most of your cores)")
    p.add_argument("--chunk", type=int, default=100_000, help="frames per work unit")
    p.add_argument("--ephe-path", default=None, help="Swiss .se1 directory")
    p.add_argument("--jpl-file", default=None, help="JPL DE441 .bsp file")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    from kalachakra.ephemeris import global_state
    if not global_state.ephemeris_available():
        print("ERROR: pyswisseph not installed.", file=sys.stderr)
        return 2
    backend = global_state.configure_from_args(ephe_path=args.ephe_path,
                                               jpl_file=args.jpl_file)
    from kalachakra.ephemeris.calendar import parse_datetime
    from kalachakra.local_autoencoder.sky_cache import build_sky_cache

    start_jd = parse_datetime(args.start)
    end_jd = parse_datetime(args.end)
    if end_jd <= start_jd:
        print("ERROR: --end must be after --start.", file=sys.stderr)
        return 2
    print(f"ephemeris backend: {backend}")
    build_sky_cache(args.out, start_jd, end_jd, cadence_hours=args.cadence_hours,
                    ephe_path=args.ephe_path, jpl_file=args.jpl_file,
                    workers=args.workers, chunk=args.chunk)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
