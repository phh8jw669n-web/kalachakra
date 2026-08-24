#!/usr/bin/env python3
"""
Phase 3 (v2) — train the Spherical Autoencoder at FULL MESH on Apple MPS.

Identical to scripts/train.py in every respect except the model: it uses the
node-chunked autoencoder (kalachakra.models.autoencoder_v2), which applies the
spatial geodesic conv and temporal FNO in slices over the node axis. This is what
makes ``--nodes 122880`` runnable on Metal: the v1 forward folds batch x nodes
into a single >2-billion-element tensor, which MPS cannot index
("MPSGraph does not support tensor dims larger than INT_MAX"). Chunking produces
a bit-for-bit identical result (nodes are independent within both ops), so model
capacity and quality are unchanged — see --node-chunk.

The store, data path, optimizer, schedule, loss, and checkpoint format are all
unchanged, so:
  * you do NOT need to regenerate data — point --store at your existing data/full;
  * checkpoints are interchangeable with the v1 loaders.

Requires:  pip install "kalachakra[train]"

Example (full mesh on an M4 Max):
    python scripts/train_v2.py --store data/full \
        --nodes 122880 --window 64 --stride 32 --batch 2 \
        --hidden 128 --blocks 3 --modes 32 --node-chunk 4096 \
        --epochs 1 --save-every 500 --checkpoints checkpoints/full
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kalachakra import constants as C                              # noqa: E402
from kalachakra.models.autoencoder_v2 import DEFAULT_NODE_CHUNK    # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--store", type=Path, default=Path("data/store"),
                   help="ephemeris store dir (auto-generated if missing)")
    p.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    p.add_argument("--start-date", default="2024-01-01",
                   help="start of the auto-generated real window (UTC)")
    p.add_argument("--frames", type=int, default=2048,
                   help="frames to generate if the store is missing")
    p.add_argument("--ephe-path", default=None,
                   help="Swiss .se1 directory, DE431 (enables the full timeline)")
    p.add_argument("--jpl-file", default=None,
                   help="DE441 .bsp file for the JPL backend (full timeline)")
    p.add_argument("--nodes", type=int, default=512,
                   help="observer nodes (mesh + model use the SAME grid)")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--latent", type=int, default=C.LATENT_DIM)
    p.add_argument("--modes", type=int, default=16, help="Fourier modes")
    p.add_argument("--blocks", type=int, default=2)
    p.add_argument("--knn", type=int, default=7)
    p.add_argument("--node-chunk", type=int, default=DEFAULT_NODE_CHUNK,
                   help="nodes processed per slice in the spatial/temporal ops. "
                        "Exact (result-invariant) memory/indexing knob: lower if "
                        "you hit OOM, raise for speed. Does not change the model.")
    p.add_argument("--quantize", action="store_true",
                   help="train the AE + hierarchical residual VQ (tokenizer)")
    p.add_argument("--window", type=int, default=48, help="temporal window frames")
    p.add_argument("--stride", type=int, default=24)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=25, help="steps between saves")
    p.add_argument("--resume", type=Path, default=None)
    return p.parse_args(argv)


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
        print("ERROR: pyswisseph not installed. Run `pip install pyswisseph`.",
              file=sys.stderr)
        raise SystemExit(2)

    mode = global_state.configure_from_args(ephe_path=args.ephe_path,
                                            jpl_file=args.jpl_file)
    print(f"  ephemeris backend: {mode}")
    import numpy as np
    start_frame = int(timeline.jd_to_frame(parse_datetime(args.start_date)))
    print(f"Generating {args.frames:,} real frames from "
          f"{format_jd(timeline.frame_to_jd(start_frame))} -> {args.store}")
    chunk = 512
    written = 0
    while written < args.frames:
        n = min(chunk, args.frames - written)
        idx = np.arange(start_frame + written, start_frame + written + n)
        frames = global_state.global_state_batch(timeline.frame_to_jd(idx))
        store.write_chunk(start_frame + written, frames.astype(np.float32))
        written += n
    print(f"  generated {written:,} frames in {len(store.chunks())} chunks")
    return store


def main(argv=None) -> int:
    args = parse_args(argv)

    import time as _time  # noqa: F401

    from torch.utils.data import DataLoader

    from kalachakra.data.dataset import EphemerisStream, StreamConfig
    from kalachakra.grid.geodesic import fibonacci_sphere
    from kalachakra.models.autoencoder import AutoencoderConfig
    from kalachakra.models.autoencoder_v2 import SphericalAutoencoderV2
    from kalachakra.models.spherical_conv import build_knn
    from kalachakra.training.checkpoint import save_model
    from kalachakra.training.optim import OptimConfig
    from kalachakra.training.trainer import TrainConfig, Trainer

    store = ensure_store(args)

    grid = fibonacci_sphere(args.nodes)
    neighbors = build_knn(grid, k=args.knn)
    cfg = AutoencoderConfig(n_nodes=args.nodes, hidden=args.hidden,
                            latent=args.latent, fourier_modes=args.modes,
                            knn=args.knn, n_blocks=args.blocks)
    if args.quantize:
        # The tokenizer (hierarchical residual VQ) is a later-phase module. On a
        # checkout that only has the base autoencoder it will be absent; fail with
        # a clear message rather than a cryptic import error. Plain training (the
        # default) never touches these modules.
        try:
            from kalachakra.models.quantized_autoencoder_v2 import (
                QuantizedSphericalAutoencoderV2,
            )
            from kalachakra.models.rvq import RVQConfig
        except ModuleNotFoundError as exc:
            print(f"ERROR: --quantize needs the tokenizer modules "
                  f"(kalachakra.models.rvq / quantized_autoencoder), which are not "
                  f"present in this checkout ({exc.name}). Drop --quantize to train "
                  f"the plain autoencoder, or add the tokenizer modules first.",
                  file=sys.stderr)
            return 2
        rvq_cfg = RVQConfig(dim=args.latent)
        model = QuantizedSphericalAutoencoderV2(cfg, neighbors, rvq_cfg,
                                                node_chunk=args.node_chunk)
    else:
        model = SphericalAutoencoderV2(cfg, neighbors, node_chunk=args.node_chunk)
    n_params = sum(p.numel() for p in model.parameters())

    stream = EphemerisStream(
        store, grid,
        StreamConfig(window_frames=args.window, window_stride=args.stride,
                     node_subsample=None),
    )
    loader = DataLoader(stream, batch_size=args.batch, num_workers=args.workers)

    tcfg = TrainConfig(optim=OptimConfig(optimizer="lion", lr=args.lr,
                                         restart_period=500),
                       log_every=args.log_every,
                       micro_checkpoint_seconds=1e18)
    trainer = Trainer(model, tcfg, checkpoint_dir=args.checkpoints)
    if args.resume:
        trainer.load(args.resume)
        print(f"Resumed from {args.resume} at step {trainer.step}")

    print(f"\nTraining (v2, node-chunk={args.node_chunk}) on {trainer.device} | "
          f"{n_params:,} params | nodes={args.nodes} latent={args.latent} "
          f"window={args.window} blocks={args.blocks}")
    print(f"Saving to {args.checkpoints}/  (every {args.save_every} steps)\n")

    def save_all(tag: str):
        trainer.save_micro()
        for name in (f"model_{tag}.pt", "model_latest.pt"):
            if args.quantize:
                from kalachakra.training.checkpoint import save_quantized_model
                save_quantized_model(args.checkpoints / name, model, cfg, neighbors,
                                     rvq_cfg, grid_xyz=grid.xyz,
                                     extra={"step": trainer.step})
            else:
                save_model(args.checkpoints / name, model, cfg, neighbors,
                           grid_xyz=grid.xyz,
                           extra={"step": trainer.step, "start_date": args.start_date})

    t0 = time.time()
    stop = False
    for epoch in range(args.epochs):
        if stop:
            break
        for e, lons in loader:
            parts = trainer.train_step(e, lons)
            if trainer.step % args.log_every == 0 or trainer.step == 1:
                lr = trainer.scheduler.get_last_lr()[0]
                rate = trainer.step / max(time.time() - t0, 1e-9)
                print(f"  epoch {epoch} step {trainer.step:5d}  "
                      f"loss={parts['total']:.4f}  geo={parts['geodesic']:.4f}  "
                      f"spec={parts['spectral']:.4f}  lr={lr:.2e}  "
                      f"({rate:.2f} steps/s)")
            if trainer.step % args.save_every == 0:
                save_all(f"step_{trainer.step:06d}")
                print(f"    saved checkpoint at step {trainer.step}")
            if args.max_steps and trainer.step >= args.max_steps:
                stop = True
                break

    save_all("final")
    trainer.save_micro()
    print(f"\nDone. {trainer.step} steps in {time.time() - t0:.1f}s.")
    print(f"Final model saved: {args.checkpoints / 'model_final.pt'}"
          + ("  (quantized: use with build_index.py --quantized-checkpoint)"
             if args.quantize else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
