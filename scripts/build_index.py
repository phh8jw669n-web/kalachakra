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
    p.add_argument("--rarity-min", type=float, default=0.0,
                   help="sparse full-scale mode: keep only tier-1/tier-2 rows with "
                        "rarity >= this (0 = dense in-memory build, the default). "
                        ">0 streams a two-pass build with bounded memory so the "
                        "fine tiers stay storable at full scale; tier-3 stays dense.")
    p.add_argument("--block-days", type=int, default=1,
                   help="sparse mode: days of frames held in memory per rollup flush")
    p.add_argument("--ephe-path", default=None)
    p.add_argument("--jpl-file", default=None)
    return p.parse_args(argv)


def _infer_window(model, grid, idx):
    """Run one contiguous window of frames through projection + tokenization.

    Returns (jds, macro, micro, leaf, z, potential, shear); all arrays are
    (T, n_nodes[, ...]). Shared by the dense and sparse build paths.
    """
    import torch
    jds = timeline.frame_to_jd(idx)
    fields = [spatial.project(global_state.global_state_frame(float(j)), float(j), grid)
              for j in jds]
    e = np.stack(fields, 0).reshape(len(idx), grid.n_nodes, -1)     # (T,N,F)
    e_t = torch.from_numpy(e.astype(np.float32)).unsqueeze(0)       # (1,T,N,F)
    macro, micro, leaf, quant = model.tokenize(e_t)
    macro = macro[0].cpu().numpy(); micro = micro[0].cpu().numpy()
    leaf = leaf[0].cpu().numpy(); z = quant[0].cpu().numpy()        # (T,N,64)
    potential = np.linalg.norm(z, axis=-1)                          # (T,N)
    shear = (np.abs(np.gradient(potential, axis=0)) if len(idx) > 1
             else np.zeros_like(potential))                         # (T,N)
    return jds, macro, micro, leaf, z, potential, shear


def main(argv=None) -> int:
    args = parse_args(argv)
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
          f"{format_jd(timeline.frame_to_jd(start_frame))}"
          + (f"  [sparse: rarity >= {args.rarity_min}]" if args.rarity_min > 0 else ""))

    if args.rarity_min > 0:
        return _build_sparse_streaming(args, model, grid, h3cells, lat_deg, lng_deg,
                                       start_frame)

    all_cols: dict[str, list] = {k: [] for k in
        ("jd", "frame", "node", "lat", "lng", "h3", "macro", "micro",
         "leaf", "potential", "shear", "latent")}
    rarity_model = RarityModel(RVQConfig().n_leaf)

    processed = 0
    while processed < args.frames:
        w = min(args.window, args.frames - processed)
        idx = np.arange(start_frame + processed, start_frame + processed + w)
        jds, macro, micro, leaf, z, potential, shear = _infer_window(model, grid, idx)

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


def _flush_rollups(store, jd, pot, shear, leaf, rar, n_nodes,
                   lat_deg, lng_deg, h3cells, rarity_min, stats):
    """Write tier-2 (hourly, sparse by bucket max rarity) + tier-3 (daily, dense)
    for one in-memory block of frames. Reuses the exact mipmap reductions."""
    n_frames = jd.shape[0]
    FPH = mipmap.FRAMES_PER_HOUR
    starts_h = np.arange(0, n_frames, FPH)
    starts_d = np.arange(0, n_frames, mipmap.FRAMES_PER_DAY)
    out_h = {k: [] for k in ("jd", "node", "lat", "lng", "h3",
                             "max_potential", "peak_shear", "archetype", "rarity")}
    out_d = {k: [] for k in ("jd", "node", "lat", "lng", "h3", "max_potential",
                             "mean_potential", "peak_shear", "rarity",
                             "anomaly_count", "archetype")}
    for node in range(n_nodes):
        # -- tier 2: hourly, kept only where the bucket's max rarity clears the bar
        roll = mipmap.hourly_rollup(pot[:, node], shear[:, node], leaf[:, node])
        hr = mipmap.bucket_max(rar[:, node], FPH)
        keep = hr >= rarity_min
        k = int(keep.sum())
        if k:
            nb = roll["max_potential"].shape[0]
            hj = jd[starts_h][:nb]
            out_h["jd"].append(hj[keep])
            out_h["node"].append(np.full(k, node, np.int32))
            out_h["lat"].append(np.full(k, lat_deg[node], np.float32))
            out_h["lng"].append(np.full(k, lng_deg[node], np.float32))
            out_h["h3"].append(np.full(k, h3cells[node], np.int64))
            out_h["max_potential"].append(roll["max_potential"][keep].astype(np.float32))
            out_h["peak_shear"].append(roll["peak_shear"][keep].astype(np.float32))
            out_h["archetype"].append(roll["archetype"][keep].astype(np.int32))
            out_h["rarity"].append(hr[keep].astype(np.float32))
        # -- tier 3: daily, dense base layer (always kept)
        rd = mipmap.daily_rollup(pot[:, node], shear[:, node], rar[:, node], leaf[:, node])
        nbd = rd["max_potential"].shape[0]
        out_d["jd"].append(jd[starts_d][:nbd])
        out_d["node"].append(np.full(nbd, node, np.int32))
        out_d["lat"].append(np.full(nbd, lat_deg[node], np.float32))
        out_d["lng"].append(np.full(nbd, lng_deg[node], np.float32))
        out_d["h3"].append(np.full(nbd, h3cells[node], np.int64))
        out_d["max_potential"].append(rd["max_potential"].astype(np.float32))
        out_d["mean_potential"].append(rd["mean_potential"].astype(np.float32))
        out_d["peak_shear"].append(rd["peak_shear"].astype(np.float32))
        out_d["rarity"].append(rd["max_rarity"].astype(np.float32))
        out_d["anomaly_count"].append(rd["anomaly_count"].astype(np.float32))
        out_d["archetype"].append(rd["archetype"].astype(np.int32))
    if out_h["jd"]:
        colsh = {kk: np.concatenate(v) for kk, v in out_h.items()}
        store.write_hourly(colsh); stats["t2"] += len(colsh["jd"])
    colsd = {kk: np.concatenate(v) for kk, v in out_d.items()}
    store.write_daily(colsd); stats["t3"] += len(colsd["jd"])


def _build_sparse_streaming(args, model, grid, h3cells, lat_deg, lng_deg, start_frame):
    """Full-scale build with bounded memory (blueprint §3-4, rarity premise §2/§6).

    Two passes: (1) accumulate the deep-time token PMF; (2) score each frame's
    rarity and STREAM writes. The fine tiers (tier-1 native, tier-2 hourly) keep
    only rows at/above ``--rarity-min`` — the rare configurations that are the
    entire point of the Rarity Index — while tier-3 (daily) stays dense as the
    always-available base layer. Memory is bounded by one rollup block, not the
    timeline length, so this is the path that actually reaches full scale.
    """
    from kalachakra.models.rvq import RVQConfig
    n_nodes = args.nodes
    n_leaf = RVQConfig().n_leaf
    block_len = max(1, args.block_days) * mipmap.FRAMES_PER_DAY
    rarity_min = args.rarity_min

    def windows():
        processed = 0
        while processed < args.frames:
            w = min(args.window, args.frames - processed)
            idx = np.arange(start_frame + processed, start_frame + processed + w)
            yield idx, _infer_window(model, grid, idx)
            processed += w

    # -- PASS 1: global token distribution (bounded: one window at a time) -----
    print("  pass 1/2: accumulating deep-time token distribution...")
    rarity_model = RarityModel(n_leaf)
    seen = 0
    for idx, out in windows():
        rarity_model.update(out[3].ravel())      # out[3] == leaf
        seen += len(idx)
    print(f"    counted {seen:,} frames over {int(rarity_model.total):,} tokens")

    # -- PASS 2: score + stream-write -----------------------------------------
    print("  pass 2/2: scoring + streaming writes...")
    store = ParquetTokenStore(args.out)
    stats = {"t1": 0, "t2": 0, "t3": 0, "total": 0}
    buf = {k: [] for k in ("jd", "potential", "shear", "leaf", "rarity")}
    buf_n = [0]
    t1 = {k: [] for k in ("jd", "frame", "node", "lat", "lng", "h3", "macro",
                          "micro", "leaf", "potential", "shear", "latent", "rarity")}

    def flush_t1(force=False):
        if not t1["jd"] or (not force and sum(len(a) for a in t1["node"]) < 200_000):
            return
        cols = {k: (np.concatenate(v) if k != "latent" else np.concatenate(v, axis=0))
                for k, v in t1.items()}
        store.write_frames(cols); stats["t1"] += len(cols["jd"])
        for v in t1.values():
            v.clear()

    def flush_block():
        if not buf["jd"]:
            return
        _flush_rollups(store, np.concatenate(buf["jd"]),
                       np.concatenate(buf["potential"], axis=0),
                       np.concatenate(buf["shear"], axis=0),
                       np.concatenate(buf["leaf"], axis=0),
                       np.concatenate(buf["rarity"], axis=0),
                       n_nodes, lat_deg, lng_deg, h3cells, rarity_min, stats)
        for v in buf.values():
            v.clear()
        buf_n[0] = 0

    for idx, (jds, macro, micro, leaf, z, potential, shear) in windows():
        w = len(idx)
        rar = rarity_model.rarity(leaf).astype(np.float32)     # (w, N)
        stats["total"] += w * n_nodes
        # tier-1 sparse (per window, so latent never accumulates over the timeline)
        ti_idx, node_idx = np.nonzero(rar >= rarity_min)
        if ti_idx.size:
            t1["jd"].append(jds[ti_idx])
            t1["frame"].append(idx[ti_idx].astype(np.int64))
            t1["node"].append(node_idx.astype(np.int32))
            t1["lat"].append(lat_deg[node_idx].astype(np.float32))
            t1["lng"].append(lng_deg[node_idx].astype(np.float32))
            t1["h3"].append(h3cells[node_idx])
            t1["macro"].append(macro[ti_idx, node_idx].astype(np.int16))
            t1["micro"].append(micro[ti_idx, node_idx].astype(np.int16))
            t1["leaf"].append(leaf[ti_idx, node_idx].astype(np.int32))
            t1["potential"].append(potential[ti_idx, node_idx].astype(np.float32))
            t1["shear"].append(shear[ti_idx, node_idx].astype(np.float32))
            t1["latent"].append(z[ti_idx, node_idx].astype(np.float32))
            t1["rarity"].append(rar[ti_idx, node_idx])
        flush_t1()
        # reduced arrays for the rollups (no latent -> bounded to one block)
        buf["jd"].append(jds); buf["potential"].append(potential.astype(np.float32))
        buf["shear"].append(shear.astype(np.float32))
        buf["leaf"].append(leaf.astype(np.int32)); buf["rarity"].append(rar)
        buf_n[0] += w
        if buf_n[0] >= block_len:
            flush_block()
        print(f"  scored {min(stats['total'] // n_nodes, args.frames):,}/"
              f"{args.frames} frames")

    flush_t1(force=True)
    flush_block()
    pct = 100.0 * stats["t1"] / max(stats["total"], 1)
    print(f"\nSparse index ready at {args.out}. "
          f"tier-1: {stats['t1']:,} rows ({pct:.3f}% of {stats['total']:,} "
          f"frame-nodes, rarity >= {rarity_min}); tier-2: {stats['t2']:,}; "
          f"tier-3 (dense): {stats['t3']:,}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
