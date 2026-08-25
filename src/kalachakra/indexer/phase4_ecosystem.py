"""Phase 4 - Domain 5: the archetype ecosystem (PRD page 5).

Treats the 4096 tokens as one closed spherical ecosystem that fights, follows and
balances itself. Purely-temporal relations come from DuckDB over Parquet;
graph-dependent ones reuse the co-occurrence matrices accumulated during the
Phase-2 sweep (DuckDB has no notion of the mesh topology).

* Transition Lineage - Markov state-transition probabilities (node token[t] ->
  token[t+1]); the reliable decay / shatter successor of each archetype.
* Global Exclusion - strongest negative correlation between tokens' daily global
  activation volumes (mutually forbidden geometries).
* Spatial Symbiosis - adjacency-halo: the boundary token most conditionally
  present around a token's clusters.
* Antipodal Resonance - the token most systematically forced onto the geographic
  antipode to balance the tensor field.
"""

from __future__ import annotations

import time

import numpy as np

from .phase3_temporal import _connect, _daily_matrix

#: The per-token scalar columns produced by Domain 5. In --lite mode they are
#: emitted as NULL (and the relation side-tables left empty) so the master-DB
#: schema is byte-for-byte the same shape the UI frontend expects.
DOMAIN5_SCALAR_KEYS = (
    "transition_top_to", "transition_top_prob",
    "symbiosis_top_token", "symbiosis_top_prob",
    "antipode_top_token", "antipode_top_prob",
    "exclusion_top_token", "exclusion_top_corr",
)


def empty_phase4(codebook_size: int):
    """Domain-5 placeholders for --lite runs: NULL scalars + empty relation graphs.

    Keeps every Domain-5 column present (as NULL) and every relation table present
    (empty) so downstream SQL against the dossier never errors in lite mode.
    """
    profiles = {t: dict.fromkeys(DOMAIN5_SCALAR_KEYS, None)
                for t in range(codebook_size)}
    relations = {"transitions": [], "exclusion": [], "symbiosis": [], "antipode": []}
    return profiles, relations


def transition_lineage(con, act_glob: str, top_k: int, logger=None):
    """Markov transitions from node token sequences; returns (top_per_token, rows)."""
    t0 = time.time()
    q = f"""
    WITH seq AS (
      SELECT node_id, frame_ord, token_id,
             LAG(token_id) OVER (PARTITION BY node_id ORDER BY frame_ord) AS prev
      FROM read_parquet('{act_glob}')
    )
    SELECT prev AS from_t, token_id AS to_t, COUNT(*) AS c
    FROM seq WHERE prev IS NOT NULL AND prev != token_id
    GROUP BY prev, token_id
    """
    rows = con.execute(q).fetchall()
    totals: dict[int, int] = {}
    for ft, _tt, c in rows:
        totals[int(ft)] = totals.get(int(ft), 0) + int(c)
    trans = []                                   # (from, to, prob)
    per_from: dict[int, list] = {}
    for ft, tt, c in rows:
        p = int(c) / totals[int(ft)]
        trans.append((int(ft), int(tt), p))
        per_from.setdefault(int(ft), []).append((int(tt), p))
    top_per_token = {}
    for ft, lst in per_from.items():
        lst.sort(key=lambda x: -x[1])
        top_per_token[ft] = lst[:top_k]
    if logger:
        logger.info(f"[P4] transition matrix: {len(rows)} nonzero edges over "
                    f"{len(per_from)} source tokens in {time.time() - t0:.2f}s")
    return top_per_token, trans


def global_exclusion(M, toks, top_k: int, logger=None):
    """Strongest negative daily-volume correlations (mutually exclusive states)."""
    t0 = time.time()
    if M is None or M.shape[0] < 3:
        return {}, []
    Z = M - M.mean(axis=0, keepdims=True)
    sd = Z.std(axis=0)                            # population std (ddof=0)
    keep = sd > 1e-9
    Zc = Z[:, keep] / sd[keep]
    tk = toks[keep]
    corr = (Zc.T @ Zc) / Zc.shape[0]             # Pearson r in [-1, 1]
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 0.0)
    top_neg = {}
    pairs = []
    for i in range(tk.size):
        j = int(corr[i].argmin())
        c = float(corr[i, j])
        if c < 0:
            top_neg[int(tk[i])] = [(int(tk[j]), round(c, 4))]
        order = np.argsort(corr[i])[:top_k]
        for j2 in order:
            if corr[i, j2] < 0:
                pairs.append((int(tk[i]), int(tk[j2]), round(float(corr[i, j2]), 4)))
    if logger:
        strongest = sorted(pairs, key=lambda x: x[2])[:5]
        logger.info(f"[P4] exclusion: {len(pairs)} negative pairs over {tk.size} "
                    f"tokens in {time.time() - t0:.2f}s; strongest: {strongest}")
    return top_neg, pairs


def _halo_from_cooc(cooc: np.ndarray, top_k: int, drop_self: bool):
    """Per-row top conditional-probability partners from a co-occurrence matrix."""
    C = cooc.astype(np.float64).copy()
    if drop_self:
        np.fill_diagonal(C, 0.0)
    row_tot = C.sum(axis=1)
    top = {}
    rows = []
    for a in np.nonzero(row_tot)[0]:
        probs = C[a] / row_tot[a]
        order = np.argsort(-probs)[:top_k]
        lst = [(int(b), round(float(probs[b]), 4)) for b in order if probs[b] > 0]
        if lst:
            top[int(a)] = lst
            for b, p in lst:
                rows.append((int(a), b, p))
    return top, rows


def run_phase4(cfg, acc, logger):
    """Domain-5 profiles + relation tables. ``acc`` carries the Phase-2 cooc matrices."""
    con = _connect()
    act_glob = str((cfg.parquet_dir / "activations" / "*.parquet"))
    day_glob = str((cfg.parquet_dir / "daily_volume" / "*.parquet"))

    trans_top, trans_rows = transition_lineage(con, act_glob, cfg.top_k_relations, logger)
    M, _days, toks = _daily_matrix(con, day_glob)
    excl_top, excl_rows = global_exclusion(M, toks, cfg.top_k_relations, logger)
    con.close()

    sym_top, sym_rows = _halo_from_cooc(np.asarray(acc.cooc_sym), cfg.top_k_relations,
                                        drop_self=True)
    anti_top, anti_rows = _halo_from_cooc(np.asarray(acc.cooc_anti), cfg.top_k_relations,
                                          drop_self=True)
    if logger:
        logger.info(f"[P4] symbiosis halos for {len(sym_top)} tokens; "
                    f"antipodal resonance for {len(anti_top)} tokens")

    # per-token scalar summary (top partner of each relation)
    profiles: dict = {}
    all_tokens = set(trans_top) | set(excl_top) | set(sym_top) | set(anti_top)
    for t in all_tokens:
        d = {}
        if t in trans_top and trans_top[t]:
            d["transition_top_to"] = trans_top[t][0][0]
            d["transition_top_prob"] = trans_top[t][0][1]
        if t in sym_top and sym_top[t]:
            d["symbiosis_top_token"] = sym_top[t][0][0]
            d["symbiosis_top_prob"] = sym_top[t][0][1]
        if t in anti_top and anti_top[t]:
            d["antipode_top_token"] = anti_top[t][0][0]
            d["antipode_top_prob"] = anti_top[t][0][1]
        if t in excl_top and excl_top[t]:
            d["exclusion_top_token"] = excl_top[t][0][0]
            d["exclusion_top_corr"] = excl_top[t][0][1]
        profiles[t] = d

    relations = {
        "transitions": trans_rows,          # (from, to, prob)
        "exclusion": excl_rows,             # (a, b, corr)
        "symbiosis": sym_rows,              # (token, halo_token, prob)
        "antipode": anti_rows,              # (token, antipode_token, prob)
    }
    return profiles, relations
