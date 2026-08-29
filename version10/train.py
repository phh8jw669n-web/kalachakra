#!/usr/bin/env python3
"""Train the version10 Topocentric Self-Attention colour field.

Every step draws a fresh random batch of continuous ``(lat, lon, jd)`` observer skies, feeds
the 11 topocentric body tokens (N,E,Zenith) through the micro self-attention network, and
trains it so colour distances mirror the observer-dependent sky distance (local vectors +
horizon-gated chords). The gamut-bounded head keeps colour displayable; no data files.

Example:
    python -m version10.train --steps 40000 --batch 2048 --export version10/web/weights.json
"""

from __future__ import annotations

import argparse


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--d-model", type=int, default=32)
    p.add_argument("--d-ff", type=int, default=64)
    p.add_argument("--d-head", type=int, default=32)
    p.add_argument("--blocks", type=int, default=2)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--jd-start", type=float, default=None)
    p.add_argument("--jd-end", type=float, default=None)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-min", type=float, default=1e-6)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--steps", type=int, default=40_000)
    p.add_argument("--gamma", type=float, default=0.35,
                   help="chroma scale ||d(OKLab a,b)|| = gamma*d_sky (OKLab units, ~60x < CIELab)")
    p.add_argument("--w-local", type=float, default=0.5, help="weight on 39-D local distance")
    p.add_argument("--w-rel", type=float, default=0.5, help="weight on 78-D horizon-gated chords")
    p.add_argument("--gate-k", type=float, default=3.0,
                   help="horizon-gate steepness (v10 default 3.0, softer = less spatial ringing)")
    p.add_argument("--weight-decay", type=float, default=3e-3,
                   help="AdamW weight decay on 2-D matrices only (v10.1 anti-inflation; was 1e-4)")
    p.add_argument("--tv-weight", type=float, default=0.0,
                   help="fine-scale isometry-consistency weight (v10.1.1: 0 = off; the Cartesian "
                        "head makes it unnecessary. Set >0 for optional belt-and-suspenders)")
    p.add_argument("--tv-delta", type=float, default=0.6, help="fine neighbour offset in degrees")
    p.add_argument("--tv-weight-coarse", type=float, default=0.0,
                   help="coarse-scale isometry-consistency weight (0 = off)")
    p.add_argument("--tv-delta-coarse", type=float, default=2.5,
                   help="coarse neighbour offset in degrees")
    p.add_argument("--no-qk-norm", action="store_true",
                   help="disable bounded cosine attention (legacy unbounded dot-product)")
    p.add_argument("--attn-temp-max", type=float, default=30.0,
                   help="max cosine-attention temperature (logit clamp) when qk_norm is on")
    p.add_argument("--anchor-weight", type=float, default=0.05)
    p.add_argument("--device", default="")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--out-dir", default="version10/checkpoints")
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--resume", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--export", nargs="?", const="version10/web/weights.json", default=None,
                   help="after training, export attention weights to this JSON path")
    return p.parse_args(argv)


def build_config(args):
    from version10.config import AttnConfig, DataConfig, TrainConfig, V10Config
    attn = AttnConfig(d_model=args.d_model, d_ff=args.d_ff, d_head=args.d_head,
                      n_blocks=args.blocks, qk_norm=not args.no_qk_norm,
                      attn_temp_max=args.attn_temp_max)
    data = DataConfig(batch=args.batch, seed=args.seed)
    if args.jd_start is not None:
        data.jd_start = args.jd_start
    if args.jd_end is not None:
        data.jd_end = args.jd_end
    train = TrainConfig(lr=args.lr, lr_min=args.lr_min, warmup_steps=args.warmup,
                        max_steps=args.steps, gamma=args.gamma, w_local=args.w_local,
                        w_rel=args.w_rel, gate_k=args.gate_k, weight_decay=args.weight_decay,
                        tv_weight=args.tv_weight, tv_delta_deg=args.tv_delta,
                        tv_weight_coarse=args.tv_weight_coarse,
                        tv_delta_coarse_deg=args.tv_delta_coarse,
                        anchor_weight=args.anchor_weight,
                        device=args.device, num_workers=args.workers, out_dir=args.out_dir,
                        save_every=args.save_every, log_every=args.log_every, seed=args.seed)
    return V10Config(attn=attn, data=data, train=train)


def main(argv=None) -> int:
    args = parse_args(argv)
    from version10.training import export_weights_json, train
    final = train(build_config(args), resume=args.resume)
    if args.export:
        out = export_weights_json(str(final), args.export)
        print(f"exported attention weights -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
