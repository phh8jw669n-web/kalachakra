#!/usr/bin/env python3
"""
Phase 2/3 offline inference: build the tokenized, queryable index (blueprint §7).

Streams real ephemeris frames, projects each onto the geodesic mesh, encodes and
quantizes with the hierarchical residual VQ, computes the deep-time Rarity Index,
and serializes the per-(frame,node) descriptors to partitioned Parquet (tier 1),
hourly rollups (tier 2), and daily/epochal rollups (tier 3) — the three-tier
temporal mipmap the DuckDB engine routes across and the web client queries.

Turn-key: with no store it generates a real window (Moshier or the configured
full-span backend). A trained quantized checkpoint gives meaningful tokens; with
none it builds a fresh model so you can still exercise the full data path.

Requires:  pip install "kalachakra[train]" pyarrow duckdb h3

Example:
    python scripts/build_index.py --out data/index --nodes 256 --frames 3000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kalachakra import constants as C                              # noqa: E402
from kalachakra.analysis import tokens as tk                       # noqa: E402
from kalachakra.analysis.rarity import RarityModel                 # noqa: E402
from kalachakra.ephemeris import global_state, timeline           # noqa: E402
from kalachakra.ephemeris.calendar import format_jd, parse_datetime  # noqa: E402
from kalachakra.geo import h3index                                 # noqa: E402
from kalachakra.grid import geodesic                               # noqa: E402
from kalachakra.projection import spatial                          # noqa: E402
from kalachakra.storage import mipmap                              # noqa: E402
from kalachakra.storage.parquet_store import ParquetTokenStore     # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("data/index"))
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="trained plain-AE checkpoint (wrapped with a fresh RVQ)")
    p.add_argument("--quantized-checkpoint", type=Path, default=None,
                   help="trained quantized (AE+RVQ) checkpoint — meaningful tokens")
    p.add_argument("--nodes", type=int, default=256)
    p.add_argument("--start-date", default="2024-01-01")
    p.add_argument("--frames", type=int, default=3000)
    p.add_argument("--window", type=int, default=128, help="frames per inference batch")
    p.add_argument("--ephe-path", default=None)
    p.add_argument("--jpl-file", default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    import torch
    from kalachakra.models.autoencoder import AutoencoderConfig
    from kalachakra.models.quantized_autoencoder import QuantizedSphericalAutoencoder
    from kalachakra.models.rvq import RVQConfig
    from kalachakra.models.spherical_conv import build_knn

    if not global_state.ephemeris_available():
        print("ERROR: pyswisseph not installed.", file=sys.stderr)
        return 2
    global_state.configure_from_args(ephe_path=args.ephe_path, jpl_file=args.jpl_file)

    if args.quantized_checkpoint and args.quantized_checkpoint.exists():
        from kalachakra.training.checkpoint import load_quantized_model
        model, ae_cfg, _rvq, grid_xyz = load_quantized_model(args.quantized_checkpoint)
        if grid_xyz is not None:
            grid = geodesic.Grid(
                xyz=np.asarray(grid_xyz),
                lat=np.arcsin(np.clip(np.asarray(grid_xyz)[:, 2], -1, 1)),
                lon=np.arctan2(np.asarray(grid_xyz)[:, 1], np.asarray(grid_xyz)[:, 0]))
            args.nodes = grid.n_nodes
        else:
            grid = geodesic.fibonacci_sphere(ae_cfg.n_nodes); args.nodes = ae_cfg.n_nodes
        print(f"Loaded trained quantized model from {args.quantized_checkpoint} "
              f"({args.nodes} nodes) — tokens are meaningful.")
    else:
        grid = geodesic.fibonacci_sphere(args.nodes)
        neighbors = build_knn(grid, 7)
        ae_cfg = AutoencoderConfig(n_nodes=args.nodes, hidden=64, latent=C.LATENT_DIM,
                                   fourier_modes=16, knn=7, n_blocks=2)
        model = QuantizedSphericalAutoencoder(ae_cfg, neighbors, RVQConfig()).eval()
        if args.checkpoint and args.checkpoint.exists():
            from kalachakra.training.checkpoint import load_model
            trained, _cfg, _xyz = load_model(args.checkpoint)
            model.ae.load_state_dict(trained.state_dict())
            print(f"Loaded trained encoder/decoder from {args.checkpoint}")
        else:
            print("No checkpoint: using a fresh (untrained) model — the data path "
                  "is real; tokens become meaningful once trained.")

    lat_deg = np.rad2deg(grid.lat)
    lng_deg = np.rad2deg(grid.lon)
    h3cells = h3index.cells_for_grid(lat_deg, lng_deg, h3index.BASE_RESOLUTION)

    start_frame = int(timeline.jd_to_frame(parse_datetime(args.start_date)))
    print(f"Indexing {args.frames} frames x {args.nodes} nodes from "
          f"{format_jd(timeline.frame_to_jd(start_frame))}")

    all_cols: dict[str, list] = {k: [] for k in
        ("jd", "frame", "node", "lat", "lng", "h3", "macro", "micro",
         "leaf", "potential", "shear", "latent")}
    rarity_model = RarityModel(RVQConfig().n_leaf)

    processed = 0
    while processed < args.frames:
        w = min(args.window, args.frames - processed)
        idx = np.arange(start_frame + processed, start_frame + processed + w)
        jds = timeline.frame_to_jd(idx)
        fields = [spatial.project(global_state.global_state_frame(float(j)), float(j), grid)
                  for j in jds]
        e = np.stack(fields, 0).reshape(w, args.nodes, -1)          # (T,N,F)
        e_t = torch.from_numpy(e.astype(np.float32)).unsqueeze(0)   # (1,T,N,F)

        macro, micro, leaf, quant = model.tokenize(e_t)
        macro = macro[0].cpu().numpy(); micro = micro[0].cpu().numpy()
        leaf = leaf[0].cpu().numpy(); z = quant[0].cpu().numpy()    # (T,N,64)

        potential = np.linalg.norm(z, axis=-1)                     # (T,N)
        shear = np.abs(np.gradient(potential, axis=0))             # (T,N)

        for ti in range(w):
            all_cols["jd"].append(np.full(args.nodes, jds[ti]))
            all_cols["frame"].append(np.full(args.nodes, idx[ti], dtype=np.int64))
            all_cols["node"].append(np.arange(args.nodes, dtype=np.int32))
            all_cols["lat"].append(lat_deg.astype(np.float32))
            all_cols["lng"].append(lng_deg.astype(np.float32))
            all_cols["h3"].append(h3cells)
            all_cols["macro"].append(macro[ti].astype(np.int16))
            all_cols["micro"].append(micro[ti].astype(np.int16))
            all_cols["leaf"].append(leaf[ti].astype(np.int32))
            all_cols["potential"].append(potential[ti].astype(np.float32))
            all_cols["shear"].append(shear[ti].astype(np.float32))
            all_cols["latent"].append(z[ti].astype(np.float32))
        rarity_model.update(leaf.ravel())
        processed += w
        print(f"  processed {processed}/{args.frames} frames")

    cols = {k: (np.concatenate(v) if k != "latent" else np.concatenate(v, axis=0))
            for k, v in all_cols.items()}
    cols["rarity"] = rarity_model.rarity(cols["leaf"]).astype(np.float32)

    store = ParquetTokenStore(args.out)
    n = store.write_frames(cols)
    print(f"Wrote {n:,} tier-1 rows -> {store.tier1}")

    # Tier-2 hourly rollups + tier-3 daily (epochal) rollups per node.
    _write_hourly(store, cols, args.nodes)
    _write_daily(store, cols, args.nodes)

    print(f"\nIndex ready at {args.out}. Query it with the DuckDB engine or "
          f"scripts/serve.py. Rarity spans [{cols['rarity'].min():.3f}, "
          f"{cols['rarity'].max():.3f}].")
    return 0


def _write_hourly(store: ParquetTokenStore, cols: dict, n_nodes: int) -> None:
    """Aggregate tier-1 rows into per-node hourly (150-frame) rollups."""
    frames = np.unique(cols["frame"])
    n_frames = frames.shape[0]
    pot = cols["potential"].reshape(n_frames, n_nodes)
    shear = cols["shear"].reshape(n_frames, n_nodes)
    leaf = cols["leaf"].reshape(n_frames, n_nodes)
    jd = cols["jd"].reshape(n_frames, n_nodes)[:, 0]

    starts = np.arange(0, n_frames, mipmap.FRAMES_PER_HOUR)
    out = {k: [] for k in ("jd", "node", "lat", "lng", "h3",
                           "max_potential", "peak_shear", "archetype")}
    lat = cols["lat"].reshape(n_frames, n_nodes)[0]
    lng = cols["lng"].reshape(n_frames, n_nodes)[0]
    h3c = cols["h3"].reshape(n_frames, n_nodes)[0]
    for node in range(n_nodes):
        roll = mipmap.hourly_rollup(pot[:, node], shear[:, node], leaf[:, node])
        nb = roll["max_potential"].shape[0]
        out["jd"].append(jd[starts])
        out["node"].append(np.full(nb, node, dtype=np.int32))
        out["lat"].append(np.full(nb, lat[node], dtype=np.float32))
        out["lng"].append(np.full(nb, lng[node], dtype=np.float32))
        out["h3"].append(np.full(nb, h3c[node], dtype=np.int64))
        out["max_potential"].append(roll["max_potential"].astype(np.float32))
        out["peak_shear"].append(roll["peak_shear"].astype(np.float32))
        out["archetype"].append(roll["archetype"].astype(np.int32))
    cols2 = {k: np.concatenate(v) for k, v in out.items()}
    cols2["rarity"] = np.zeros(len(cols2["jd"]), dtype=np.float32)  # placeholder col
    store.write_hourly(cols2)
    print(f"Wrote {len(cols2['jd']):,} tier-2 hourly rollup rows -> {store.tier2}")


def _write_daily(store: ParquetTokenStore, cols: dict, n_nodes: int) -> None:
    """Aggregate tier-1 rows into per-node daily (3600-frame) rollups (tier 3).

    Tier 3 carries statistical summaries plus a deep-time anomaly count (frames at
    or above the rarity threshold), so an epochal viewport scan stays bounded while
    still surfacing rare events. ``max_rarity`` is stored as the ``rarity`` column
    the DuckDB router filters and orders on.
    """
    frames = np.unique(cols["frame"])
    n_frames = frames.shape[0]
    pot = cols["potential"].reshape(n_frames, n_nodes)
    shear = cols["shear"].reshape(n_frames, n_nodes)
    leaf = cols["leaf"].reshape(n_frames, n_nodes)
    rar = cols["rarity"].reshape(n_frames, n_nodes)
    jd = cols["jd"].reshape(n_frames, n_nodes)[:, 0]
    lat = cols["lat"].reshape(n_frames, n_nodes)[0]
    lng = cols["lng"].reshape(n_frames, n_nodes)[0]
    h3c = cols["h3"].reshape(n_frames, n_nodes)[0]

    starts = np.arange(0, n_frames, mipmap.FRAMES_PER_DAY)
    out = {k: [] for k in ("jd", "node", "lat", "lng", "h3", "max_potential",
                           "mean_potential", "peak_shear", "rarity",
                           "anomaly_count", "archetype")}
    for node in range(n_nodes):
        roll = mipmap.daily_rollup(pot[:, node], shear[:, node],
                                   rar[:, node], leaf[:, node])
        nb = roll["max_potential"].shape[0]
        out["jd"].append(jd[starts][:nb])
        out["node"].append(np.full(nb, node, dtype=np.int32))
        out["lat"].append(np.full(nb, lat[node], dtype=np.float32))
        out["lng"].append(np.full(nb, lng[node], dtype=np.float32))
        out["h3"].append(np.full(nb, h3c[node], dtype=np.int64))
        out["max_potential"].append(roll["max_potential"].astype(np.float32))
        out["mean_potential"].append(roll["mean_potential"].astype(np.float32))
        out["peak_shear"].append(roll["peak_shear"].astype(np.float32))
        out["rarity"].append(roll["max_rarity"].astype(np.float32))
        out["anomaly_count"].append(roll["anomaly_count"].astype(np.float32))
        out["archetype"].append(roll["archetype"].astype(np.int32))
    cols3 = {k: np.concatenate(v) for k, v in out.items()}
    store.write_daily(cols3)
    print(f"Wrote {len(cols3['jd']):,} tier-3 daily rollup rows -> {store.tier3}")


if __name__ == "__main__":
    raise SystemExit(main())
