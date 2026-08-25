#!/usr/bin/env python3
"""
The Great Indexer - deep-time archetype profiler (CLI entry point).

Sweeps the ephemeris timeline with adaptive time-stepping, using PyTorch purely as
the physics engine, flushing activation records to compressed Parquet, aggregating
them with DuckDB, and compiling 18 mathematical profiles across five domains for
all 4096 VQ archetypes into a single queryable SQLite dossier database.

The pipeline is resumable and crash-safe: rerun the same command after any
interruption and it skips completed phases and resumes the sweep at the exact
frame it stopped (state.json + accum.npz).

Requires:  pip install "kalachakra[train,index,transducer]"  (torch + pyswisseph +
           pyarrow + duckdb + scipy);  psutil is optional (hardware telemetry).

Examples:
    # small MVP window (fast)
    python scripts/great_indexer.py --checkpoint checkpoints/v3/model_step_000025.pt \
        --start-date 2024-01-01T00:00:00Z --end-date 2024-01-11T00:00:00Z \
        --out-dir index_out

    # full 10,256-year sweep (very long; the architecture this pipeline was built for)
    python scripts/great_indexer.py --checkpoint checkpoints/v3/model_step_000025.pt \
        --full --out-dir index_out
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_args(argv=None):
    from kalachakra import constants as C
    from kalachakra.ephemeris.calendar import parse_datetime

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="checkpoints/v3/model_step_000025.pt")
    p.add_argument("--out-dir", default="index_out")
    p.add_argument("--start-date", default=None, help="UTC ISO start (default: timeline epoch)")
    p.add_argument("--end-date", default=None, help="UTC ISO end")
    p.add_argument("--days", type=float, default=None,
                   help="window length in days from --start-date (alt to --end-date)")
    p.add_argument("--full", action="store_true",
                   help="sweep the entire 10,256-year timeline (overrides dates)")
    p.add_argument("--coarse-seconds", type=float, default=3600.0)
    p.add_argument("--fine-seconds", type=float, default=float(C.VIGHATIKA_SECONDS))
    p.add_argument("--velocity-threshold", type=float, default=0.02)
    p.add_argument("--chunk-frames", type=int, default=50_000)
    p.add_argument("--calib-days", type=int, default=24)
    p.add_argument("--epoch-years", type=int, default=50)
    p.add_argument("--node-batch", type=int, default=0)
    p.add_argument("--device", default="")
    args = p.parse_args(argv)
    args._parse_datetime = parse_datetime
    args._C = C
    return args


def build_config(args):
    from kalachakra.indexer.config import IndexerConfig
    C = args._C
    cfg = IndexerConfig(
        checkpoint=args.checkpoint, out_dir=args.out_dir,
        coarse_step_seconds=args.coarse_seconds, fine_step_seconds=args.fine_seconds,
        velocity_threshold=args.velocity_threshold, chunk_frames=args.chunk_frames,
        calib_days=args.calib_days, epoch_years=args.epoch_years,
        node_batch=args.node_batch, device=args.device)
    if args.full:
        b = C.timeline_bounds()
        cfg.start_jd, cfg.end_jd = b.start_jd, b.end_jd
    elif args.start_date:
        cfg.start_jd = args._parse_datetime(args.start_date)
        if args.end_date:
            cfg.end_jd = args._parse_datetime(args.end_date)
        elif args.days:
            cfg.end_jd = cfg.start_jd + args.days
        else:
            cfg.end_jd = cfg.start_jd + 10.0
    else:
        # default MVP: 10 days from "now"
        cfg.start_jd = args._parse_datetime("now")
        cfg.end_jd = cfg.start_jd + (args.days or 10.0)
    return cfg


def main(argv=None) -> int:
    args = parse_args(argv)
    if not Path(args.checkpoint).exists():
        print(f"ERROR: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 2
    from kalachakra.indexer.pipeline import run_pipeline
    cfg = build_config(args)
    try:
        db = run_pipeline(cfg)
    except KeyboardInterrupt:
        print("\ninterrupted; rerun the same command to resume from the last chunk.",
              file=sys.stderr)
        return 130
    print(f"\nmaster dossier DB: {db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
