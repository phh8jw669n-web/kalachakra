#!/usr/bin/env python3
"""
Phase 3 — train the Spherical Autoencoder over the ephemeris stream (blueprint §4-5).

Wires the geodesic grid, the streaming dataset, the STFNO autoencoder and the
Trainer (Lion + cosine-annealing warm restarts, BF16 mixed precision, two-tier
checkpointing). Point it at a store produced by ``generate_ephemeris.py``.

Requires:  pip install "kalachakra[train]"

Example:
    python scripts/train.py --store data/ephemeris --checkpoints checkpoints \\
        --nodes 122880 --window 64 --batch 4 --max-steps 100000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kalachakra import constants as C                              # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store", type=Path, required=True, help="ephemeris store dir")
    p.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    p.add_argument("--nodes", type=int, default=C.N_SPATIAL_NODES)
    p.add_argument("--window", type=int, default=64, help="temporal window frames")
    p.add_argument("--stride", type=int, default=32)
    p.add_argument("--node-subsample", type=int, default=4096,
                   help="random node subset per window (memory control)")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--resume", type=Path, default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    import torch
    from torch.utils.data import DataLoader

    from kalachakra.data.dataset import EphemerisStream, StreamConfig
    from kalachakra.grid.geodesic import fibonacci_sphere
    from kalachakra.models.autoencoder import AutoencoderConfig, SphericalAutoencoder
    from kalachakra.models.spherical_conv import build_knn
    from kalachakra.storage.binary_store import EphemerisStore
    from kalachakra.training.trainer import TrainConfig, Trainer

    grid = fibonacci_sphere(args.nodes)
    # The model runs on the (possibly subsampled) node set the stream emits.
    active_nodes = args.node_subsample or args.nodes
    active_grid = fibonacci_sphere(active_nodes)
    neighbors = build_knn(active_grid, k=7)

    model = SphericalAutoencoder(
        AutoencoderConfig(n_nodes=active_nodes, knn=7), neighbors
    )

    store = EphemerisStore(args.store)
    stream = EphemerisStream(
        store, grid,
        StreamConfig(window_frames=args.window, window_stride=args.stride,
                     node_subsample=args.node_subsample),
    )
    loader = DataLoader(stream, batch_size=args.batch, num_workers=args.workers)

    trainer = Trainer(model, TrainConfig(), checkpoint_dir=args.checkpoints)
    if args.resume:
        trainer.load(args.resume)
        print(f"Resumed from {args.resume} at step {trainer.step}")

    print(f"Training on {trainer.device} | nodes={active_nodes} | "
          f"latent={C.LATENT_DIM} | window={args.window}")
    trainer.fit(loader, max_steps=args.max_steps)
    trainer.save_micro()
    print("Training loop exited; final checkpoint saved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
