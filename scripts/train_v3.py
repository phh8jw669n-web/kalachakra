#!/usr/bin/env python3
"""
Phase 3 (v3) — train the discrete VQ-bottleneck autoencoder and SAVE the models.

This script is self-contained for everything it OWNS: the model is the standalone
kalachakra.models.autoencoder_v3 (its own conv / FNO / ST-block / Vector
Quantizer), and the training loop, VQ-loss aggregation, codebook-perplexity
logging, and checkpointing are all implemented here — it does not use the v1/v2
trainer or model files. It reuses only the read-only shared *data + physics*
utilities (the ephemeris store reader, the geodesic mesh, and the geo+spec loss
definitions) so the physics stays a single source of truth.

Total loss:  L = L_geo + L_spec + L_VQ   (VQ = codebook + beta*commitment)
Logged per step: geo, spec, vq, perplexity (active-codebook monitor).

Full-mesh safe: pass --node-chunk / --vq-chunk to tile the ops and --grad-checkpoint
to fit the 122,880-node mesh in memory (see scripts/train_v2.py notes).

Requires:  pip install "kalachakra[train]"

Example (full mesh on an M4 Max):
    python scripts/train_v3.py --store data/full \
        --nodes 122880 --window 64 --stride 32 --batch 1 \
        --hidden 128 --blocks 3 --modes 32 --codebook 4096 --beta 0.25 \
        --node-chunk 8192 --vq-chunk 131072 --grad-checkpoint \
        --save-every 50 --log-every 5 --checkpoints checkpoints/v3
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kalachakra import constants as C                              # noqa: E402

#: projection semantics this model was trained under (see constants).
_PROJECTION_VERSION = getattr(C, "PROJECTION_VERSION", 1)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--store", type=Path, default=Path("data/store"),
                   help="ephemeris store dir (auto-generated if missing)")
    p.add_argument("--checkpoints", type=Path, default=Path("checkpoints_v3"))
    p.add_argument("--start-date", default="2024-01-01")
    p.add_argument("--frames", type=int, default=2048,
                   help="frames to generate if the store is missing")
    p.add_argument("--ephe-path", default=None)
    p.add_argument("--jpl-file", default=None)
    # model / mesh
    p.add_argument("--nodes", type=int, default=512)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--latent", type=int, default=C.LATENT_DIM)
    p.add_argument("--modes", type=int, default=16)
    p.add_argument("--blocks", type=int, default=2)
    p.add_argument("--knn", type=int, default=7)
    # VQ
    p.add_argument("--codebook", type=int, default=4096, help="number of archetypes")
    p.add_argument("--beta", type=float, default=0.25,
                   help="commitment weight inside the VQ loss")
    p.add_argument("--lambda-vq", type=float, default=0.25,
                   help="scale on the VQ loss in the total: geo+spec+lambda_vq*vq")
    p.add_argument("--ema-decay", type=float, default=0.99,
                   help="EMA decay for the codebook cluster centers")
    p.add_argument("--restart-after", type=int, default=10,
                   help="restart a code unused for this many consecutive steps")
    # full-mesh memory / indexing knobs
    p.add_argument("--node-chunk", type=int, default=8192)
    p.add_argument("--vq-chunk", type=int, default=131_072)
    p.add_argument("--grad-checkpoint", action="store_true")
    # optimization
    p.add_argument("--window", type=int, default=48)
    p.add_argument("--stride", type=int, default=24)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--empty-cache-every", type=int, default=-1,
                   help="release the accelerator's cached memory pool every N "
                        "steps to fight allocator fragmentation (the usual cause "
                        "of creeping step times on MPS). -1 = auto (10 on MPS, "
                        "off elsewhere); 0 = never.")
    p.add_argument("--resume", type=Path, default=None)
    return p.parse_args(argv)


def select_device():
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_store(args):
    from kalachakra.ephemeris import global_state, timeline
    from kalachakra.ephemeris.calendar import format_jd, parse_datetime
    from kalachakra.storage.binary_store import EphemerisStore

    store = EphemerisStore(args.store)
    if store.chunks():
        print(f"Using existing store {args.store} "
              f"({sum(c.n_frames for c in store.chunks()):,} frames)")
        return store
    if not global_state.ephemeris_available():
        print("ERROR: pyswisseph not installed.", file=sys.stderr)
        raise SystemExit(2)
    mode = global_state.configure_from_args(ephe_path=args.ephe_path,
                                            jpl_file=args.jpl_file)
    print(f"  ephemeris backend: {mode}")
    start_frame = int(timeline.jd_to_frame(parse_datetime(args.start_date)))
    print(f"Generating {args.frames:,} real frames from "
          f"{format_jd(timeline.frame_to_jd(start_frame))} -> {args.store}")
    written = 0
    while written < args.frames:
        n = min(512, args.frames - written)
        idx = np.arange(start_frame + written, start_frame + written + n)
        frames = global_state.global_state_batch(timeline.frame_to_jd(idx))
        store.write_chunk(start_frame + written, frames.astype(np.float32))
        written += n
    print(f"  generated {written:,} frames in {len(store.chunks())} chunks")
    return store


def main(argv=None) -> int:
    args = parse_args(argv)

    import torch
    from torch.utils.data import DataLoader

    from kalachakra.data.dataset import EphemerisStream, StreamConfig
    from kalachakra.grid.geodesic import fibonacci_sphere
    from kalachakra.losses.geometric import CompositeGeodesicLoss
    from kalachakra.models.autoencoder_v3 import (
        VQAutoencoderV3, VQAutoencoderV3Config, build_knn,
    )
    from kalachakra.training.optim import OptimConfig, build_optimizer, build_scheduler

    store = ensure_store(args)

    grid = fibonacci_sphere(args.nodes)
    neighbors = build_knn(grid.xyz, args.knn)
    cfg = VQAutoencoderV3Config(
        n_nodes=args.nodes, in_features=C.LOCAL_FIELD_WIDTH, hidden=args.hidden,
        latent=args.latent, fourier_modes=args.modes, knn=args.knn,
        n_blocks=args.blocks, codebook_size=args.codebook, commitment_beta=args.beta,
        ema_decay=args.ema_decay, restart_after=args.restart_after,
        node_chunk=args.node_chunk, vq_chunk=args.vq_chunk,
        grad_checkpoint=args.grad_checkpoint,
    )
    device = select_device()
    model = VQAutoencoderV3(cfg, neighbors).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    stream = EphemerisStream(
        store, grid,
        StreamConfig(window_frames=args.window, window_stride=args.stride,
                     node_subsample=None),
    )
    loader = DataLoader(stream, batch_size=args.batch, num_workers=args.workers)

    ocfg = OptimConfig(optimizer="lion", lr=args.lr, restart_period=500)
    optimizer = build_optimizer(model.parameters(), ocfg)
    scheduler = build_scheduler(optimizer, ocfg)
    criterion = CompositeGeodesicLoss()          # geodesic + spectral terms
    amp_dtype = torch.bfloat16

    args.checkpoints.mkdir(parents=True, exist_ok=True)
    step = 0
    if args.resume and Path(args.resume).exists():
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["state_dict"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        step = ck.get("step", 0)
        print(f"Resumed from {args.resume} at step {step}")

    print(f"\nTraining (v3 VQ, codebook={args.codebook}, beta={args.beta}, "
          f"node-chunk={args.node_chunk}, grad_checkpoint={args.grad_checkpoint}) "
          f"on {device} | {n_params:,} params | nodes={args.nodes} "
          f"latent={args.latent} window={args.window} blocks={args.blocks}")
    print(f"Saving to {args.checkpoints}/  (every {args.save_every} steps)\n")

    def save(tag: str):
        payload = {
            "format": "kalachakra-vqmodel-v3",
            "projection_version": _PROJECTION_VERSION,
            "config": asdict(cfg),
            "neighbors": np.asarray(neighbors, dtype=np.int64),
            "grid_xyz": np.asarray(grid.xyz, dtype=np.float64),
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
        }
        torch.save(payload, args.checkpoints / f"model_{tag}.pt")
        torch.save(payload, args.checkpoints / "model_latest.pt")
        torch.save(payload, args.checkpoints / "micro_latest.pt")

    # Periodic accelerator cache release — the usual cure for creeping step times
    # on MPS (cached-allocator fragmentation), a no-op on CPU.
    ec_every = args.empty_cache_every
    if ec_every < 0:
        ec_every = 10 if device.type == "mps" else 0
    if device.type not in ("mps", "cuda"):
        ec_every = 0                                  # nothing to release on CPU

    def _empty_cache():
        if device.type == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

    if ec_every:
        print(f"  (releasing {device.type} cache every {ec_every} steps)")

    t0 = time.time()
    stop = False
    for epoch in range(args.epochs):
        if stop:
            break
        for e, _lons in loader:
            e = e.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                recon, _z, _idx, vq_loss = model(e)
            # geo + spec computed in fp32 outside autocast (FFT precision).
            recon_f = recon.float().unflatten(-1, (-1, 5))
            target_f = e.float().unflatten(-1, (-1, 5))
            gs_total, parts = criterion(recon_f, target_f)
            # total = L_geo + L_spec + lambda_vq * L_VQ
            total = gs_total + args.lambda_vq * vq_loss.float()
            total.backward()
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step(step)
            step += 1
            if ec_every and step % ec_every == 0:
                _empty_cache()

            if step % args.log_every == 0 or step == 1:
                lr = scheduler.get_last_lr()[0]
                elapsed = max(time.time() - t0, 1e-9)
                rate = step / elapsed
                sps = elapsed / step               # seconds per step
                print(f"  epoch {epoch} step {step:5d}  "
                      f"loss={float(total.detach()):.4f}  "
                      f"geo={float(parts['geodesic'].detach()):.4f}  "
                      f"spec={float(parts['spectral'].detach()):.4f}  "
                      f"vq={float(vq_loss.detach()):.4f}  "
                      f"ppl={float(model.last_perplexity):.1f}  "
                      f"lr={lr:.2e}  ({rate:.5f} steps/s, {sps:.1f}s/step)")
            if step % args.save_every == 0:
                save(f"step_{step:06d}")
                print(f"    saved checkpoint at step {step}")
            if args.max_steps and step >= args.max_steps:
                stop = True
                break

    save("final")
    print(f"\nDone. {step} steps in {time.time() - t0:.1f}s.")
    print(f"Final model saved: {args.checkpoints / 'model_final.pt'} "
          f"(codebook {args.codebook}x{args.latent} in state_dict['vq.codebook']).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
