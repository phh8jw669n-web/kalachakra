#!/usr/bin/env python3
"""Train the version6 SIREN isometric colour field.

Every step draws a fresh random batch of continuous ``(lat, lon, jd)`` skies, builds
the 33-D topocentric tensor, and trains the SIREN so colour distances mirror physical
distances. No dataset, no grid, no ephemeris data files.

Example:
    python -m version6.train --steps 40000 --batch 2048 \
        --export version6/web/weights.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # siren
    p.add_argument("--hidden", type=int, default=48)
    p.add_argument("--hidden-layers", type=int, default=2)
    p.add_argument("--omega0", type=float, default=30.0)
    # data
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--jd-start", type=float, default=None)
    p.add_argument("--jd-end", type=float, default=None)
    # optim / loss
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-min", type=float, default=1e-6)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--steps", type=int, default=40_000)
    p.add_argument("--color-scale", type=float, default=20.0,
                   help="target ||dLab|| = color_scale * ||dSky|| (1.0 = pure isometry)")
    p.add_argument("--anchor-weight", type=float, default=0.05)
    p.add_argument("--device", default="")
    p.add_argument("--workers", type=int, default=0)
    # io
    p.add_argument("--out-dir", default="version6/checkpoints")
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--resume", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--export", nargs="?", const="version6/web/weights.json", default=None,
                   help="after training, export SIREN weights to this JSON path")
    return p.parse_args(argv)


def build_config(args):
    from version6.config import DataConfig, SirenConfig, TrainConfig, V6Config
    siren = SirenConfig(hidden=args.hidden, hidden_layers=args.hidden_layers,
                        omega0=args.omega0)
    data = DataConfig(batch=args.batch, seed=args.seed)
    if args.jd_start is not None:
        data.jd_start = args.jd_start
    if args.jd_end is not None:
        data.jd_end = args.jd_end
    train = TrainConfig(lr=args.lr, lr_min=args.lr_min, warmup_steps=args.warmup,
                        max_steps=args.steps, color_scale=args.color_scale,
                        anchor_weight=args.anchor_weight, device=args.device,
                        num_workers=args.workers, out_dir=args.out_dir,
                        save_every=args.save_every, log_every=args.log_every, seed=args.seed)
    return V6Config(siren=siren, data=data, train=train)


def main(argv=None) -> int:
    args = parse_args(argv)
    from version6.training import export_weights_json, train
    final = train(build_config(args), resume=args.resume)
    if args.export:
        out = export_weights_json(str(final), args.export)
        print(f"exported SIREN weights -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
