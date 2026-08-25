"""Phase 3 - Domain 4: temporal waveforms via DuckDB over Parquet (PRD page 4).

PyTorch is disconnected; DuckDB streams the compressed activation records with
minimal memory. Three profiles:

* Persistence Baseline - run-length encoding along each node's temporal axis to
  get mean dwell (consecutive frames a node holds a token) and an exponential
  half-life estimate.
* Harmonic Periodicity - FFT / power spectral density of each token's *daily*
  global activation volume (strictly one-frame-per-day points, per the PRD, to
  avoid 24 s micro-frame aliasing); reports the dominant period.
* Epoch Clustering - Fano factor (variance/mean) of activation counts over
  50-year epochs; ~1 = steady background, >>1 = generational burst.
"""

from __future__ import annotations

import time

import numpy as np

from .. import constants as C


def _connect():
    import duckdb
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    return con


def persistence_baseline(con, act_glob: str, logger=None) -> dict:
    """Mean dwell (frames) + half-life per token via gaps-and-islands RLE."""
    t0 = time.time()
    q = f"""
    WITH ordered AS (
      SELECT node_id, frame_ord, token_id,
             row_number() OVER (PARTITION BY node_id ORDER BY frame_ord) AS rn
      FROM read_parquet('{act_glob}')
    ),
    islands AS (
      SELECT node_id, token_id, frame_ord,
             rn - row_number() OVER (PARTITION BY node_id, token_id ORDER BY frame_ord) AS grp
      FROM ordered
    ),
    runs AS (
      SELECT token_id, COUNT(*) AS run_len
      FROM islands GROUP BY node_id, token_id, grp
    )
    SELECT token_id, AVG(run_len) AS mean_dwell, COUNT(*) AS n_runs,
           MAX(run_len) AS max_dwell
    FROM runs GROUP BY token_id
    """
    rows = con.execute(q).fetchall()
    out = {}
    for tok, mean_dwell, n_runs, max_dwell in rows:
        md = float(mean_dwell)
        half_life = float(np.log(2.0) * md) if md > 0 else 0.0
        out[int(tok)] = {
            "dwell_frames_mean": round(md, 3),
            "dwell_frames_max": int(max_dwell),
            "dwell_half_life_frames": round(half_life, 3),
            "n_runs": int(n_runs),
        }
    if logger:
        logger.info(f"[P3] persistence RLE over Parquet: {len(out)} tokens, "
                    f"{time.time() - t0:.2f}s")
    return out


def _daily_matrix(con, day_glob: str):
    """Dense (n_days, K_present) volume matrix + token id list, from daily_volume."""
    rows = con.execute(
        f"SELECT day_index, token_id, SUM(volume) v "
        f"FROM read_parquet('{day_glob}') GROUP BY day_index, token_id").fetchall()
    if not rows:
        return None, None, None
    days = np.array(sorted({r[0] for r in rows}), dtype=np.int64)
    toks = np.array(sorted({r[1] for r in rows}), dtype=np.int64)
    day_ix = {d: i for i, d in enumerate(days.tolist())}
    tok_ix = {t: i for i, t in enumerate(toks.tolist())}
    M = np.zeros((days.size, toks.size), dtype=np.float64)
    for d, t, v in rows:
        M[day_ix[d], tok_ix[t]] = v
    return M, days, toks


def harmonic_periodicity(M, days, toks, min_samples=16, logger=None) -> dict:
    """Dominant FFT period (days) of each token's daily activation-volume series."""
    t0 = time.time()
    out = {}
    if M is None or days.size < min_samples:
        if logger:
            logger.info(f"[P3] harmonic: only {0 if days is None else days.size} daily "
                        f"points (< {min_samples}); skipping FFT.")
        return out
    # dense day grid so the FFT sees a uniform sampling interval (1 day)
    full = np.arange(days.min(), days.max() + 1)
    idx = days - days.min()
    dense = np.zeros((full.size, toks.size))
    dense[idx] = M
    dense -= dense.mean(axis=0, keepdims=True)
    spec = np.abs(np.fft.rfft(dense, axis=0)) ** 2               # PSD
    freqs = np.fft.rfftfreq(full.size, d=1.0)                    # cycles/day
    spec[0] = 0.0                                                # drop DC
    for j, tok in enumerate(toks.tolist()):
        col = spec[:, j]
        if not np.any(col > 0):
            continue
        k = int(col.argmax())
        f = freqs[k]
        period = float(1.0 / f) if f > 0 else 0.0
        power_frac = float(col[k] / (col.sum() + 1e-12))
        out[int(tok)] = {
            "dominant_period_days": round(period, 3),
            "dominant_power_fraction": round(power_frac, 4),
        }
    if logger:
        sample = list(out.items())[:5]
        logger.info(f"[P3] harmonic FFT done for {len(out)} tokens in "
                    f"{time.time() - t0:.2f}s")
        for tok, d in sample:
            logger.info(f"[P3]   token {tok}: peak period {d['dominant_period_days']} d "
                        f"(power {d['dominant_power_fraction']})")
    return out


def epoch_clustering(M, days, toks, epoch_years=50, logger=None) -> dict:
    """Fano factor (var/mean) of per-epoch activation counts per token."""
    t0 = time.time()
    out = {}
    if M is None:
        return out
    epoch_days = epoch_years * C.DAYS_PER_YEAR
    epoch_of = ((days - days.min()) / epoch_days).astype(np.int64)
    n_epochs = int(epoch_of.max()) + 1
    binned = np.zeros((n_epochs, toks.size))
    for i, e in enumerate(epoch_of.tolist()):
        binned[e] += M[i]
    mean = binned.mean(axis=0)
    var = binned.var(axis=0)
    fano = np.divide(var, mean, out=np.zeros_like(var), where=mean > 0)
    for j, tok in enumerate(toks.tolist()):
        out[int(tok)] = {
            "fano_factor": round(float(fano[j]), 4),
            "n_epochs": int(n_epochs),
            "epoch_years": int(epoch_years),
        }
    if logger:
        logger.info(f"[P3] epoch clustering ({n_epochs} x {epoch_years}y bins) for "
                    f"{len(out)} tokens in {time.time() - t0:.2f}s")
    return out


def run_phase3(cfg, logger) -> dict:
    """All Domain-4 profiles keyed by token id (merged sub-dicts)."""
    con = _connect()
    act_glob = str((cfg.parquet_dir / "activations" / "*.parquet"))
    day_glob = str((cfg.parquet_dir / "daily_volume" / "*.parquet"))
    persistence = persistence_baseline(con, act_glob, logger)
    M, days, toks = _daily_matrix(con, day_glob)
    harmonic = harmonic_periodicity(M, days, toks, cfg.fft_min_samples, logger)
    epochs = epoch_clustering(M, days, toks, cfg.epoch_years, logger)
    con.close()

    merged: dict = {}
    for src in (persistence, harmonic, epochs):
        for tok, d in src.items():
            merged.setdefault(tok, {}).update(d)
    return merged
