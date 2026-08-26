#!/usr/bin/env python3
"""
Micro-benchmark the Local Sky Autoencoder training step (train_v4).

Times ONLY the model forward+backward+optimizer step on synthetic tensors -- no
ephemeris, no data loading -- so you can find the fastest (batch size, precision)
on your accelerator in seconds instead of waiting through training warmup. Uses a
proper device sync so the numbers are real (MPS/CUDA dispatch is asynchronous).

Compares AMP on vs off: fp16 autocast on MPS can silently fall back some ops to the
CPU, which makes it *slower* -- this tells you immediately which to use.

Example:
    python scripts/bench_v4.py --batches 512,2048,8192,16384,32768
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="", help="'' -> auto (mps/cuda/cpu)")
    p.add_argument("--batches", default="512,2048,8192,16384",
                   help="comma-separated batch sizes to sweep")
    p.add_argument("--iters", type=int, default=25, help="timed steps per config")
    p.add_argument("--warmup", type=int, default=8, help="warmup steps per config")
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--dim-ff", type=int, default=512)
    p.add_argument("--amp", choices=["both", "on", "off"], default="both")
    return p.parse_args(argv)


def _sync(device) -> None:
    import torch
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def _empty_cache(device) -> None:
    import torch
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def _one_step(model, opt, x, tg, w, device, use_amp, amp_dtype) -> None:
    import torch

    from kalachakra.local_autoencoder.losses import physics_weighted_mse
    opt.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
        recon, _ok = model(x)
        loss = physics_weighted_mse(recon.float(), tg, w)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()


def bench_config(device, cfg, batch: int, use_amp: bool, amp_dtype,
                 iters: int, warmup: int) -> float:
    """Return mean seconds/step for one (batch, precision) config."""
    import torch

    from kalachakra.local_autoencoder.model import build_model

    model = build_model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    x = torch.randn(batch, 10, 8, device=device)
    tg = torch.randn(batch, 11, 8, device=device)
    w = torch.ones(batch, 11, 8, device=device)

    for _ in range(warmup):
        _one_step(model, opt, x, tg, w, device, use_amp, amp_dtype)
    _sync(device)
    t0 = time.time()
    for _ in range(iters):
        _one_step(model, opt, x, tg, w, device, use_amp, amp_dtype)
    _sync(device)
    dt = (time.time() - t0) / iters
    del model, opt, x, tg, w
    _empty_cache(device)
    return dt


def main(argv=None) -> int:
    args = parse_args(argv)
    import torch

    from kalachakra.local_autoencoder.config import ModelConfig
    from kalachakra.local_autoencoder.training import select_device

    device = select_device(args.device)
    amp_modes = {"both": [False, True], "on": [True], "off": [False]}[args.amp]
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float16
    batches = [int(b) for b in args.batches.split(",") if b.strip()]
    cfg = ModelConfig(d_model=args.d_model, nhead=args.nhead,
                      num_layers=args.layers, dim_feedforward=args.dim_ff)

    print(f"device={device}  d_model={args.d_model} heads={args.nhead} "
          f"layers={args.layers} ff={args.dim_ff}  amp_dtype={amp_dtype}")
    print(f"{'batch':>8} {'amp':>5} {'ms/step':>10} {'samp/s':>12}")
    print("-" * 40)
    for amp in amp_modes:
        use_amp = amp and device.type in ("cuda", "mps")
        for b in batches:
            try:
                dt = bench_config(device, cfg, b, use_amp, amp_dtype,
                                  args.iters, args.warmup)
                print(f"{b:>8} {('on' if amp else 'off'):>5} "
                      f"{dt * 1000:>10.1f} {b / dt:>12,.0f}")
            except RuntimeError as exc:                 # e.g. OOM at a large batch
                print(f"{b:>8} {('on' if amp else 'off'):>5}   FAILED: "
                      f"{str(exc).splitlines()[0][:40]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
