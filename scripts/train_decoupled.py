#!/usr/bin/env python3
"""
Train the decoupled projection engine (Sky Encoder + Earth Lens Decoder).

Self-supervised training over the continuous 10,256-year timeline: the Sky Encoder
compresses the ten-body ephemeris state into a 512-D global tension vector and the
Earth Lens Decoder maps that vector plus arbitrary lat/lon to an OKLab colour. The
objective combines a geometric-interference contrastive loss (Sky Encoder), a
terrestrial geodesic-smoothness loss relaxed at planetary culmination boundaries
(Earth Lens), and a temporal-continuity loss -- no human labels.

Optimised with AdamW + cosine-annealing-with-warmup + gradient clipping, AMP on
CUDA / Apple-Silicon MPS, and resumable checkpoints.

Requires:  pip install "kalachakra[train]"  (torch + pyswisseph). Deep-time spans
need the Swiss / DE441 files (pass --ephe-path / --jpl-file); otherwise restrict the
range with --timeline-start / --timeline-end (Moshier covers ~3000 BCE-3000 CE).

Example:
    python scripts/train_decoupled.py --steps 20000 --batch 8 \
        --timeline-start 0001-01-01 --timeline-end 3000-01-01 \
        --out-dir checkpoints/decoupled
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # data / timeline
    p.add_argument("--timeline-start", default=None, help="UTC ISO sweep start")
    p.add_argument("--timeline-end", default=None, help="UTC ISO sweep end")
    p.add_argument("--temporal-len", type=int, default=3)
    p.add_argument("--stride-seconds", type=float, default=3600.0)
    p.add_argument("--points-per-frame", type=int, default=1024)
    p.add_argument("--samples-per-epoch", type=int, default=4096)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--ephe-path", default=None)
    p.add_argument("--jpl-file", default=None)
    # sky encoder
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--sky-layers", type=int, default=4)
    p.add_argument("--tension-dim", type=int, default=512)
    p.add_argument("--grad-checkpoint", action="store_true")
    # earth lens
    p.add_argument("--num-fourier", type=int, default=64)
    p.add_argument("--fourier-scale", type=float, default=8.0)
    p.add_argument("--earth-hidden", type=int, default=256)
    p.add_argument("--earth-blocks", type=int, default=4)
    p.add_argument("--activation", choices=["gauss", "sine"], default="gauss")
    # optimisation
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    p.add_argument("--device", default="")
    # loss weights
    p.add_argument("--w-geometric", type=float, default=1.0)
    p.add_argument("--w-terrestrial", type=float, default=0.5)
    p.add_argument("--w-temporal", type=float, default=0.25)
    # io
    p.add_argument("--out-dir", default="checkpoints/decoupled")
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def build_config(args):
    from kalachakra.decoupled_engine.config import (
        DataConfig, EarthLensConfig, EngineConfig, SkyEncoderConfig, TrainConfig,
    )
    from kalachakra.ephemeris.calendar import parse_datetime

    data = DataConfig(temporal_len=args.temporal_len,
                      stride_seconds=args.stride_seconds,
                      points_per_frame=args.points_per_frame,
                      samples_per_epoch=args.samples_per_epoch, seed=args.seed)
    if args.timeline_start:
        data.start_jd = parse_datetime(args.timeline_start)
    if args.timeline_end:
        data.end_jd = parse_datetime(args.timeline_end)

    sky = SkyEncoderConfig(d_model=args.d_model, nhead=args.nhead,
                           num_layers=args.sky_layers, tension_dim=args.tension_dim,
                           grad_checkpoint=args.grad_checkpoint)
    earth = EarthLensConfig(tension_dim=args.tension_dim, num_fourier=args.num_fourier,
                            fourier_scale=args.fourier_scale, hidden=args.earth_hidden,
                            n_blocks=args.earth_blocks, activation=args.activation)
    train = TrainConfig(lr=args.lr, weight_decay=args.weight_decay,
                        warmup_steps=args.warmup, max_steps=args.steps,
                        batch_size=args.batch, grad_clip=args.grad_clip,
                        w_geometric=args.w_geometric, w_terrestrial=args.w_terrestrial,
                        w_temporal=args.w_temporal, amp=not args.no_amp,
                        device=args.device, out_dir=args.out_dir,
                        save_every=args.save_every, log_every=args.log_every,
                        seed=args.seed)
    return EngineConfig(sky=sky, earth=earth, data=data, train=train)


def main(argv=None) -> int:
    args = parse_args(argv)
    from kalachakra.ephemeris import global_state
    if not global_state.ephemeris_available():
        print("ERROR: pyswisseph not installed (required for training).",
              file=sys.stderr)
        return 2
    from kalachakra.decoupled_engine.training import train
    cfg = build_config(args)
    train(cfg, num_workers=args.workers, ephe_path=args.ephe_path,
          jpl_file=args.jpl_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
