"""Phase 1 - Domain 1: tensor physics of the 4096 archetypes (PRD page 2).

Four profiles extracted directly from the model before the temporal sweep:

* True Magnitude - mean L2 norm of the PRE-quantization continuous latents per
  token (a cosine VQ codebook is unit-normalized, so static magnitudes are all
  1.0; the real "tension intensity" is measured from a calibration pass).
* Dimensional Variance - variance across the 64 dims of each codebook vector
  (low = blunt/smooth pressure wave, high = jagged micro-stress).
* Anomaly Rank - cosine isolation: mean off-diagonal similarity of each token to
  the whole codebook; low mean similarity => rare, isolated geometric state.
* Principal Component Dominance - SVD of the codebook; each token's absolute
  projection weight onto the top axes, and which axis dominates (PC1 = deep slow
  background frequency ... PC3 = localized high-frequency ripple).
"""

from __future__ import annotations

import time

import numpy as np

from .model_io import project_fields, tokenize_batch


def calibrate_magnitude(model, grid, device, calib_jds, codebook_size, batch, logger=None):
    """Mean pre-quantization ||z|| per token over a small calibration sweep."""
    sum_mag = np.zeros(codebook_size, dtype=np.float64)
    cnt = np.zeros(codebook_size, dtype=np.int64)
    jds = list(calib_jds)
    for s in range(0, len(jds), batch):
        fields = project_fields(grid, jds[s:s + batch])                 # (b,N,50)
        tokens, mags = tokenize_batch(model, fields, device, want_latent_norm=True)
        flat_t = tokens.reshape(-1)
        flat_m = mags.reshape(-1)
        sum_mag += np.bincount(flat_t, weights=flat_m, minlength=codebook_size)
        cnt += np.bincount(flat_t, minlength=codebook_size)
    mean_mag = np.zeros(codebook_size, dtype=np.float64)
    seen = cnt > 0
    mean_mag[seen] = sum_mag[seen] / cnt[seen]
    if logger:
        logger.info(f"[P1] magnitude calibration over {len(jds)} frames: "
                    f"{int(seen.sum())}/{codebook_size} tokens observed, "
                    f"mean||z|| in [{mean_mag[seen].min():.3f}, {mean_mag[seen].max():.3f}]")
    return mean_mag, cnt


def run_phase1(model, grid, device, calib_jds, batch, logger=None) -> dict:
    """Compute all four Domain-1 profiles; returns ``{token_id: {..physics..}}``."""
    t0 = time.time()
    cb = model.vq.codebook.detach().cpu().numpy().astype(np.float64)     # (K, 64) unit norm
    k, d = cb.shape

    # -- Dimensional variance -------------------------------------------------
    dim_variance = cb.var(axis=1)                                        # (K,)

    # -- Anomaly rank (cosine isolation) -------------------------------------
    cb_unit = cb / (np.linalg.norm(cb, axis=1, keepdims=True) + 1e-12)
    sim = cb_unit @ cb_unit.T                                            # (K, K) cosine
    np.fill_diagonal(sim, 0.0)
    mean_sim = sim.sum(axis=1) / (k - 1)                                 # mean off-diag
    # rank: 0 = most isolated (lowest mean similarity), 100 = most typical
    order = np.argsort(mean_sim, kind="stable")
    isolation_pctile = np.empty(k)
    isolation_pctile[order] = 100.0 * (np.arange(1, k + 1) / k)
    isolation_pctile = 100.0 - isolation_pctile                         # low sim -> high isolation

    # -- Principal component dominance (SVD) ---------------------------------
    cb_centered = cb - cb.mean(axis=0, keepdims=True)
    # economy SVD: rows = tokens; V rows are the principal axes.
    _u, _s, vt = np.linalg.svd(cb_centered, full_matrices=False)
    axes = vt[:3]                                                        # top-3 PCs (3, 64)
    proj = np.abs(cb_centered @ axes.T)                                  # (K, 3) |weights|
    pc_dominant = proj.argmax(axis=1) + 1                                # 1..3

    # -- True magnitude via calibration --------------------------------------
    mean_mag, _cnt = calibrate_magnitude(
        model, grid, device, calib_jds, k, batch, logger)
    seen = mean_mag > 0
    mag_pctile = np.zeros(k)
    if seen.any():
        vals = mean_mag[seen]
        # percentile among observed tokens
        for i in np.where(seen)[0]:
            mag_pctile[i] = 100.0 * float(np.mean(vals <= mean_mag[i]))

    profiles = {}
    for t in range(k):
        profiles[t] = {
            "magnitude": round(float(mean_mag[t]), 5),
            "magnitude_percentile": round(float(mag_pctile[t]), 2),
            "dim_variance": round(float(dim_variance[t]), 6),
            "anomaly_mean_similarity": round(float(mean_sim[t]), 5),
            "anomaly_isolation_percentile": round(float(isolation_pctile[t]), 2),
            "pc_dominant": int(pc_dominant[t]),
            "pc1_weight": round(float(proj[t, 0]), 5),
            "pc2_weight": round(float(proj[t, 1]), 5),
            "pc3_weight": round(float(proj[t, 2]), 5),
        }
    if logger:
        logger.info(f"[P1] Domain-1 physics complete for {k} tokens in "
                    f"{time.time() - t0:.2f}s (dim={d}, top-3 PCs of codebook).")
        top_iso = np.argsort(mean_sim)[:3].tolist()
        logger.info(f"[P1] most isolated tokens (lowest mean cosine): {top_iso}")
    return profiles
