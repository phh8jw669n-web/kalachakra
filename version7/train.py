#!/usr/bin/env python3
"""Train the version7 regional/city-grid SIREN colour field.

Replaces v6's infinite random sampler with a structured, high-density global dataset — major
metropolitan hubs + a regional lat/lon lattice — sampled across the ~10,000-year timeline.
The bounded, soft-clamped L*a*b* head (reused from version6) keeps every colour inside the
human-perceivable gamut (L* in 0..100, a*/b* bounded), permanently eliminating the neon-cyan
/ white clipping artifacts.

On finish it exports everything the texture-mapping frontend needs:
``weights.json`` (SIREN), ``cities.json`` (markers) and ``manifest.json`` (grid + timeline).

Example:
    python -m version7.train --steps 40000 --batch 2048 --export version7/web
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
    # render grid
    p.add_argument("--grid-w", type=int, default=180)
    p.add_argument("--grid-h", type=int, default=90)
    # data / sampler
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--city-frac", type=float, default=0.35)
    p.add_argument("--grid-frac", type=float, default=0.45)
    p.add_argument("--grid-step", type=float, default=5.0, help="regional lattice spacing (deg)")
    p.add_argument("--jitter", type=float, default=2.0, help="spatial jitter around nodes (deg)")
    p.add_argument("--jd-start", type=float, default=None)
    p.add_argument("--jd-end", type=float, default=None)
    # optim / loss
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-min", type=float, default=1e-6)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--steps", type=int, default=40_000)
    p.add_argument("--color-scale", type=float, default=20.0)
    p.add_argument("--anchor-weight", type=float, default=0.05)
    p.add_argument("--device", default="")
    p.add_argument("--workers", type=int, default=0)
    # io
    p.add_argument("--out-dir", default="version7/checkpoints")
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--resume", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--export", nargs="?", const="version7/web", default=None,
                   help="after training, export weights.json + cities.json + manifest.json here")
    return p.parse_args(argv)


def build_config(args):
    from version6.config import SirenConfig

    from version7.config import DataConfig, GridConfig, TrainConfig, V7Config
    siren = SirenConfig(hidden=args.hidden, hidden_layers=args.hidden_layers, omega0=args.omega0)
    grid = GridConfig(width=args.grid_w, height=args.grid_h)
    data = DataConfig(batch=args.batch, city_frac=args.city_frac, grid_frac=args.grid_frac,
                      grid_step_deg=args.grid_step, jitter_deg=args.jitter, seed=args.seed)
    if args.jd_start is not None:
        data.jd_start = args.jd_start
    if args.jd_end is not None:
        data.jd_end = args.jd_end
    train = TrainConfig(lr=args.lr, lr_min=args.lr_min, warmup_steps=args.warmup,
                        max_steps=args.steps, color_scale=args.color_scale,
                        anchor_weight=args.anchor_weight, device=args.device,
                        num_workers=args.workers, out_dir=args.out_dir,
                        save_every=args.save_every, log_every=args.log_every, seed=args.seed)
    return V7Config(siren=siren, grid=grid, data=data, train=train)


def main(argv=None) -> int:
    args = parse_args(argv)
    from version7.training import export_manifest, export_weights_json, train
    cfg = build_config(args)
    final = train(cfg, resume=args.resume)
    if args.export:
        out = export_weights_json(str(final), str(Path(args.export) / "weights.json"))
        export_manifest(cfg, args.export)
        print(f"exported SIREN weights -> {out}")
        print(f"exported cities.json + manifest.json -> {args.export}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
