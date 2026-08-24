#!/usr/bin/env python3
"""
Archetype Profiling — Concept B: "Absolute Weather Map".

Extracts the *empirical* physics and planetary drivers of the discrete VQ tokens
of a trained ST-FNO v3 model (checkpoints/v3/model_step_000025.pt), for the top-5
most active tokens on Earth for the current date, and writes ``web/dossiers.json``.

Everything is a direct numpy/torch extraction from the model and the ephemeris —
no hand-authored labels, no archetype names. Each token is described by:

  1. Physics (from the codebook):   L2 magnitude + its percentile, and the
                                     structural variance across the 64 latent dims.
  2. Spatial footprint (today):     current global coverage and the latitudinal
                                     affinity (mean/std) of the nodes it occupies.
  3. Planetary DNA (history):       the input feature most correlated with the
                                     token's prevalence over a random historical
                                     sweep, mapped back to its celestial body.

Note on magnitude: this model uses a *cosine* VQ, so the codebook rows are unit-
normalized (||z|| == 1 by construction) and carry no intensity signal. So the
token "magnitude" is measured empirically: the mean L2 norm of the *pre*-
quantization continuous latents ``z_e`` over all nodes assigned to that token
today. Its percentile ranks that intensity among all currently-active tokens.
Structural variance is still taken across the 64 codebook dimensions.

The 50 input features are the per-node local field E(t,s): for each of the 10
bodies (Sun, Moon, Mercury, Venus, Mars, Jupiter, Saturn, Rahu, Ketu, Ayanamsha)
five components [cos(theta)cos(h), sin(theta)cos(h), sin(h), cos(dphi), sin(dphi)].
Feature index ``f`` therefore belongs to body ``f // 5``.

Requires:  pip install "kalachakra[train,transducer]"   (torch + pyswisseph)

Example:
    python scripts/profile_archetypes.py --checkpoint checkpoints/v3/model_step_000025.pt
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TOP_K = 5


def select_device():
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model_and_grid(checkpoint: str, device):
    """Load a v3 checkpoint and rebuild the model + its geodesic grid."""
    import torch

    from kalachakra.grid.geodesic import Grid
    from kalachakra.models.autoencoder_v3 import VQAutoencoderV3, VQAutoencoderV3Config

    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if ck.get("format") != "kalachakra-vqmodel-v3":
        raise ValueError(f"not a v3 checkpoint (format={ck.get('format')!r}): {checkpoint}")
    cfg = VQAutoencoderV3Config(**ck["config"])
    neighbors = np.asarray(ck["neighbors"], dtype=np.int64)
    model = VQAutoencoderV3(cfg, neighbors)
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    xyz = np.asarray(ck["grid_xyz"], dtype=np.float64)
    lat = np.arcsin(np.clip(xyz[:, 2], -1.0, 1.0))
    lon = np.arctan2(xyz[:, 1], xyz[:, 0])
    return model, cfg, Grid(xyz=xyz, lat=lat, lon=lon)


def project_day(grid, jd: float) -> np.ndarray:
    """Local field E(t,s) for one Julian Day, shape ``(N, 50)`` float64."""
    from kalachakra.ephemeris import global_state
    from kalachakra.projection import spatial

    g = global_state.global_state_frame(float(jd))
    return spatial.project(g, float(jd), grid).reshape(grid.n_nodes, -1)


def tokenize_fields(model, fields: np.ndarray, device) -> np.ndarray:
    """Discrete token ids for a batch of daily fields.

    ``fields`` is ``(B, N, 50)``; a trivial length-1 time axis is added so the
    ST-FNO encoder sees ``(B, 1, N, 50)``. Returns ``(B, N)`` int64 token ids.
    """
    import torch

    e = torch.from_numpy(fields[:, None].astype(np.float32)).to(device)   # (B,1,N,50)
    idx = model.tokenize(e)                                               # (B,1,N)
    return idx[:, 0].detach().cpu().numpy().astype(np.int64)


def encode_and_tokenize(model, fields: np.ndarray, device):
    """Continuous pre-quantization latent norm AND token id for every node.

    Returns ``(node_magnitude[N] float64, token_ids[N] int64)`` for a single
    daily field ``(N, 50)``. The magnitude is ``||z_e||_2`` of the *continuous*
    latent before the codebook lookup — the token's empirical "tension intensity",
    which (unlike the unit-normalized codebook vector) actually varies.
    """
    import torch

    e = torch.from_numpy(fields[None, None].astype(np.float32)).to(device)  # (1,1,N,50)
    with torch.no_grad():
        z = model.encode(e)                       # (1,1,N,64) continuous, pre-VQ
        _zq, idx, _l, _p = model.vq(z)            # eval() -> no EMA mutation
        mag = z[0, 0].norm(dim=1)                 # (N,)
    return mag.detach().cpu().numpy().astype(np.float64), \
        idx[0, 0].detach().cpu().numpy().astype(np.int64)


# ---------------------------------------------------------------------------
# 1. Physics profile — straight from the codebook
# ---------------------------------------------------------------------------
def codebook_physics(model):
    """Per-token (magnitude, magnitude_percentile, structural_variance)."""
    cb = model.vq.codebook.detach().cpu().numpy().astype(np.float64)      # (K, D)
    norms = np.linalg.norm(cb, axis=1)                                    # (K,)
    variance = cb.var(axis=1)                                             # (K,) across D
    # Percentile of each magnitude among all tokens (fraction with norm <= this).
    order = np.argsort(norms, kind="stable")
    pct = np.empty_like(norms)
    pct[order] = 100.0 * (np.arange(1, len(norms) + 1) / len(norms))
    unit_norm = bool(np.allclose(norms, 1.0, atol=1e-3))
    return cb, norms, variance, pct, unit_norm


# ---------------------------------------------------------------------------
# 3. Planetary DNA — correlation of token prevalence with input features
# ---------------------------------------------------------------------------
def pearson_columns(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pearson r between each column of ``x`` (D, F) and vector ``y`` (D,)."""
    xc = x - x.mean(axis=0, keepdims=True)
    yc = y - y.mean()
    denom = np.sqrt((xc ** 2).sum(axis=0) * (yc ** 2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (xc * yc[:, None]).sum(axis=0) / denom
    return r                                                              # (F,) may hold nan


def planetary_drivers(model, grid, top_tokens, n_days, n_years, batch, seed, device):
    """Correlate each top token's daily prevalence with the 50 input features.

    Samples ``n_days`` random Julian Days from the last ``n_years`` years, runs a
    forward pass per day, and for each top token correlates its per-day global
    coverage against the per-day node-mean of every input feature. Returns, per
    token, ``(feature_index, correlation, per_feature_corr)``.
    """
    from kalachakra.ephemeris.calendar import parse_datetime

    n = grid.n_nodes
    k = model.cfg.codebook_size
    jd_now = parse_datetime("now")
    rng = np.random.default_rng(seed)
    days = np.sort(rng.uniform(jd_now - n_years * 365.25, jd_now, size=n_days))

    coverage = np.zeros((n_days, len(top_tokens)), dtype=np.float64)      # (D, 5)
    feat_mean = np.zeros((n_days, 50), dtype=np.float64)                  # (D, 50)
    tok_arr = np.asarray(top_tokens, dtype=np.int64)

    for s in range(0, n_days, batch):
        e = min(s + batch, n_days)
        fields = np.stack([project_day(grid, j) for j in days[s:e]], axis=0)  # (b,N,50)
        feat_mean[s:e] = fields.mean(axis=1)                              # (b,50)
        toks = tokenize_fields(model, fields, device)                    # (b,N)
        for di in range(e - s):
            counts = np.bincount(toks[di], minlength=k).astype(np.float64)
            coverage[s + di] = counts[tok_arr] / n

    out = {}
    for j, t in enumerate(top_tokens):
        r = pearson_columns(feat_mean, coverage[:, j])                   # (50,)
        if np.all(np.isnan(r)):
            fidx, corr = -1, 0.0
        else:
            fidx = int(np.nanargmax(np.abs(r)))
            corr = float(r[fidx])
        out[t] = (fidx, corr, r)
    return out, days


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def build_dossiers(checkpoint, n_days, n_years, batch, seed, device):
    from kalachakra.ephemeris import global_state
    from kalachakra.ephemeris.bodies import ENTITIES
    from kalachakra.ephemeris.calendar import format_jd, parse_datetime

    if not global_state.ephemeris_available():
        raise RuntimeError("pyswisseph is required (`pip install \"kalachakra[transducer]\"`).")
    global_state.auto_configure()

    t0 = time.time()
    model, cfg, grid = load_model_and_grid(checkpoint, device)
    n = grid.n_nodes
    print(f"Loaded v3 model on {device}: {n:,} nodes, codebook={cfg.codebook_size}, "
          f"latent={cfg.latent}")

    # 1. Structural variance from the codebook (magnitude is measured empirically
    #    below, since this cosine VQ stores unit-norm codebook vectors).
    _cb, _norms, variance, _pct, unit_norm = codebook_physics(model)
    if unit_norm:
        print("Note: codebook is unit-normalized (cosine VQ); magnitude is measured as "
              "the mean ||z_e|| of the PRE-quantization latents in each token cluster.")

    # 2. Spatial footprint for "now": continuous latent norm + token per node.
    jd_now = parse_datetime("now")
    now_str = format_jd(jd_now)
    print(f"Tokenizing the globe for now = {now_str} ...")
    fields_now = project_day(grid, jd_now)                               # (N,50)
    node_mag, tokens_now = encode_and_tokenize(model, fields_now, device)  # (N,), (N,)
    counts_now = np.bincount(tokens_now, minlength=cfg.codebook_size)
    # Empirical per-token "tension intensity": mean pre-quant magnitude of the
    # nodes assigned to each token today.
    sum_mag = np.bincount(tokens_now, weights=node_mag, minlength=cfg.codebook_size)
    active = counts_now > 0
    mean_mag = np.zeros(cfg.codebook_size, dtype=np.float64)
    mean_mag[active] = sum_mag[active] / counts_now[active]
    active_mags = mean_mag[active]                                       # ranking population
    top_tokens = [int(t) for t in np.argsort(counts_now)[::-1] if counts_now[t] > 0][:TOP_K]
    print(f"Top {len(top_tokens)} active tokens now: {top_tokens}")
    lat_deg = np.rad2deg(grid.lat)

    # 3. Planetary DNA over a random historical sweep.
    print(f"Sweeping {n_days} random days over the last {n_years} years "
          f"(batch={batch}) for planetary correlation ...")
    drivers, days = planetary_drivers(
        model, grid, top_tokens, n_days, n_years, batch, seed, device)

    dossiers = {}
    for t in top_tokens:
        mask = tokens_now == t
        cov_pct = 100.0 * float(mask.sum()) / n
        lats = lat_deg[mask]
        fidx, corr, _r = drivers[t]
        body = ENTITIES[fidx // 5].name if fidx >= 0 else "Indeterminate"
        driver = (f"{body} (Feature Index {fidx})" if fidx >= 0
                  else "Indeterminate (no variance)")
        # percentile of this token's intensity among all currently-active tokens
        mag_pct = 100.0 * float(np.mean(active_mags <= mean_mag[t]))
        dossiers[str(t)] = {
            "id": int(t),
            "physics": {
                "magnitude": round(float(mean_mag[t]), 4),
                "magnitude_percentile": round(mag_pct, 2),
                "variance": round(float(variance[t]), 6),
            },
            "spatial": {
                "current_coverage_percent": round(cov_pct, 2),
                "mean_latitude_deg": round(float(lats.mean()), 2),
                "std_latitude_deg": round(float(lats.std()), 2),
            },
            "planetary_driver": driver,
            "planetary_driver_correlation": round(float(corr), 4),
        }

    meta = {
        "date_utc": now_str,
        "julian_day": round(float(jd_now), 5),
        "n_nodes": int(n),
        "codebook_size": int(cfg.codebook_size),
        "codebook_unit_normalized": unit_norm,
        "magnitude_source": "mean_pre_quantization_latent_norm",
        "history_days_sampled": int(n_days),
        "history_span_years": int(n_years),
        "history_jd_range": [round(float(days.min()), 3), round(float(days.max()), 3)],
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    return dossiers, meta


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="checkpoints/v3/model_step_000025.pt")
    p.add_argument("--out", default=str(ROOT / "web" / "dossiers.json"))
    p.add_argument("--days", type=int, default=100,
                   help="random historical days to sample (keep 100-200 for the MVP)")
    p.add_argument("--years", type=int, default=1000, help="look-back span in years")
    p.add_argument("--batch", type=int, default=8,
                   help="days per encoder forward pass (bounds memory)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not Path(args.checkpoint).exists():
        print(f"ERROR: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 2
    device = select_device()
    dossiers, meta = build_dossiers(args.checkpoint, args.days, args.years,
                                    args.batch, args.seed, device)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dossiers, indent=2))
    print(f"\nWrote {len(dossiers)} token dossiers -> {out}  ({meta['elapsed_seconds']}s)")
    print(json.dumps(meta, indent=2))
    for tid, d in dossiers.items():
        print(f"  token {tid:>5}: cover={d['spatial']['current_coverage_percent']:>5}%  "
              f"lat={d['spatial']['mean_latitude_deg']:>6}+-{d['spatial']['std_latitude_deg']}  "
              f"var={d['physics']['variance']:.4f}  driver={d['planetary_driver']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
