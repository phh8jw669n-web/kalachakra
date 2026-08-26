#!/usr/bin/env python3
"""Train the version5 Sky-Energy Autoencoder over the 10,256-year Monte-Carlo walk.

Each step draws one random 24-second-quantised timestamp from the full span, issues a
single ten-call ephemeris query, and reconstructs the local sky for a batch of
sphere-uniform observers — compressing it all through a 3-neuron OKLab bottleneck.

The full BCE->CE span needs the Swiss ``.se1`` files (``--ephe-path``) or JPL DE441
(``--jpl-file``); without them Moshier covers ~3000 BCE - 3000 CE, so bound the
sampler with ``--start/--end``.

Example (full span, exports the ONNX when done):
    python -m version5.train --steps 40000 --locations 2048 --workers 10 \
        --ephe-path /path/to/ephe --export version5/web/model_v5.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # data / span
    p.add_argument("--start", default=None, help="UTC ISO start ('-3101-02-18' ok)")
    p.add_argument("--end", default=None, help="UTC ISO end")
    p.add_argument("--locations", type=int, default=2048,
                   help="observer locations per timestamp (the broadcast batch)")
    # model
    p.add_argument("--d-model", type=int, default=96)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--dim-ff", type=int, default=256)
    p.add_argument("--pool", choices=["observer", "gap"], default="observer")
    # optimisation
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-min", type=float, default=1e-6,
                   help="cosine-decay floor reached at the final step")
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--steps", type=int, default=40_000)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--amp", action="store_true", help="enable autocast (off by default)")
    p.add_argument("--no-mass-weighting", action="store_true")
    p.add_argument("--device", default="")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--ephe-path", default=None, help="Swiss .se1 dir (full span)")
    p.add_argument("--jpl-file", default=None, help="JPL DE441 .bsp (alt backend)")
    # io / resume / export
    p.add_argument("--out-dir", default="version5/checkpoints")
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--resume", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--export", nargs="?", const="version5/web/model_v5.onnx",
                   default=None,
                   help="after training, export the encoder to this .onnx path "
                        "(default version5/web/model_v5.onnx)")
    return p.parse_args(argv)


def build_config(args):
    from kalachakra.ephemeris.calendar import parse_datetime

    from version5.config import DataConfig, ModelConfig, TrainConfig, V5Config
    data = DataConfig(locations_per_step=args.locations, seed=args.seed)
    if args.start:
        data.start_jd = parse_datetime(args.start)
    if args.end:
        data.end_jd = parse_datetime(args.end)
    model = ModelConfig(d_model=args.d_model, nhead=args.nhead, num_layers=args.layers,
                        dim_feedforward=args.dim_ff, pool=args.pool)
    train = TrainConfig(lr=args.lr, lr_min=args.lr_min, weight_decay=args.weight_decay,
                        warmup_steps=args.warmup, max_steps=args.steps,
                        grad_clip=args.grad_clip, amp=args.amp,
                        mass_weighting=not args.no_mass_weighting, device=args.device,
                        num_workers=args.workers, out_dir=args.out_dir,
                        save_every=args.save_every, log_every=args.log_every,
                        seed=args.seed)
    return V5Config(model=model, data=data, train=train)


def main(argv=None) -> int:
    args = parse_args(argv)
    from kalachakra.ephemeris import global_state
    if not global_state.ephemeris_available():
        print("ERROR: pyswisseph is required for version5.", file=sys.stderr)
        return 2
    from version5.training import train
    final = train(build_config(args), resume=args.resume,
                  ephe_path=args.ephe_path, jpl_file=args.jpl_file)
    if args.export:
        from version5.export_onnx import export
        export(str(final), args.export)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
