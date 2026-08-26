#!/usr/bin/env python3
"""
train_v4 — the Local Sky Autoencoder.

A purely physics-and-kinematics-driven autoencoder. Each step samples random
``(jd, lat, lon)`` points, computes the Local Sky Matrix of the ten primary bodies
(Sun..Pluto) via pyswisseph, compresses it through a self-attention Transformer into
a 3-channel OKLab colour bottleneck, and reconstructs the matrix under a
physics-weighted MSE (mass x proximity x feature). No astrology, no grids.

Runs on MPS / CUDA / CPU with AMP, AdamW + cosine-warmup, gradient clipping, rich
logging (loss + OKLab health metrics to catch mode collapse), and resumable
checkpoints.

Requires:  pip install "kalachakra[train]"  (torch + pyswisseph). Deep-time spans
need the Swiss / DE441 files; otherwise bound the range with --start / --end
(Moshier covers ~3000 BCE-3000 CE).

Example:
    python scripts/train_v4.py --steps 20000 --batch 64 \
        --start 1900-01-01 --end 2100-01-01 --out-dir checkpoints/local_sky
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # data
    p.add_argument("--start", default=None, help="UTC ISO sample-window start")
    p.add_argument("--end", default=None, help="UTC ISO sample-window end")
    # model
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--dim-ff", type=int, default=256)
    p.add_argument("--pool", choices=["observer", "gap"], default="observer")
    # optimisation
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--device", default="")
    p.add_argument("--workers", type=int, default=0,
                   help="parallel data-generation workers (physics is CPU-bound; "
                        "use 8-12 on an M-series/many-core machine for a big speedup)")
    p.add_argument("--ephe-path", default=None,
                   help="Swiss .se1 directory — required for the full BCE->CE span "
                        "(Moshier only covers ~3000 BCE-3000 CE)")
    p.add_argument("--jpl-file", default=None, help="JPL DE441 .bsp file (alt backend)")
    # io / resume
    p.add_argument("--out-dir", default="checkpoints/local_sky")
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--resume", default=None, help="checkpoint .pt to resume from")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def build_config(args):
    from kalachakra.ephemeris.calendar import parse_datetime
    from kalachakra.local_autoencoder.config import (
        DataConfig, LocalSkyConfig, ModelConfig, TrainConfig,
    )
    data = DataConfig(seed=args.seed)
    if args.start:
        data.start_jd = parse_datetime(args.start)
    if args.end:
        data.end_jd = parse_datetime(args.end)
    model = ModelConfig(d_model=args.d_model, nhead=args.nhead, num_layers=args.layers,
                        dim_feedforward=args.dim_ff, pool=args.pool)
    train = TrainConfig(lr=args.lr, weight_decay=args.weight_decay,
                        warmup_steps=args.warmup, max_steps=args.steps,
                        batch_size=args.batch, grad_clip=args.grad_clip,
                        amp=not args.no_amp, device=args.device,
                        num_workers=args.workers, out_dir=args.out_dir,
                        save_every=args.save_every, log_every=args.log_every,
                        seed=args.seed)
    return LocalSkyConfig(model=model, data=data, train=train)


def main(argv=None) -> int:
    args = parse_args(argv)
    from kalachakra.ephemeris import global_state
    if not global_state.ephemeris_available():
        print("ERROR: pyswisseph not installed (required for train_v4).",
              file=sys.stderr)
        return 2
    from kalachakra.local_autoencoder.training import train
    train(build_config(args), resume=args.resume,
          ephe_path=args.ephe_path, jpl_file=args.jpl_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
