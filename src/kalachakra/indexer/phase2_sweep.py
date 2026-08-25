"""Phase 2 - the temporal sweep: Domain-3 spatial + Domain-2 orbital (PRD page 3).

Walks the timeline with the adaptive clock, tokenizes the full mesh per frame on
the GPU, and streams two families of per-token aggregates while flushing the raw
activation records to compressed Parquet (atomically, per chunk) for Phases 3-4:

  Domain 3 (spatial):  latitudinal / polar affinity, spatial coherence &
                       dispersion (connected components), geographic drift
                       velocity (moving cluster centroid).
  Domain 2 (orbital):  multivariable attribution (ridge normal equations),
                       angular phase harmonic, orbital velocity index, solar
                       alignment angle.

Graph-dependent ecosystem statistics that DuckDB cannot express (adjacency-halo
symbiosis and antipodal co-occurrence, PRD page 5) are accumulated here too, since
they need the mesh topology in memory. Purely-temporal ecosystem stats
(transitions, exclusion) are left to Phase 4's DuckDB pass.

All streaming state is checkpointed to an atomic .npz per chunk so an interrupted
run resumes at the exact frame it stopped.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import constants as C
from ..ephemeris.calendar import format_jd, jd_to_gregorian
from .adaptive import AdaptiveClock
from .model_io import auto_node_batch, project_fields, tokenize_batch
from .state import atomic_write_bytes
from .sweep_math import (
    connected_components, great_circle_deg, latlon_of, subsolar_point,
)
from .telemetry import format_hw, hardware_snapshot

N_BODIES = C.N_BODIES
BODY_FEATS = C.LOCAL_BODY_FEATURES     # 5
IN_FEATS = C.LOCAL_FIELD_WIDTH         # 50


@dataclass
class Accum:
    """All streaming accumulators (checkpointed as one atomic .npz per chunk)."""
    K: int
    N: int
    frame_ord: int = 0
    n_frames: int = 0
    last_day: int = -10**18
    # Domain 3
    lat_sum: np.ndarray = None
    lat_sq: np.ndarray = None
    abslat_sum: np.ndarray = None
    node_count: np.ndarray = None
    frames_present: np.ndarray = None
    comp_sum: np.ndarray = None
    largest_frac_sum: np.ndarray = None
    drift_speed_sum: np.ndarray = None
    drift_lon_sum: np.ndarray = None
    drift_shear_sum: np.ndarray = None
    drift_n: np.ndarray = None
    cent_prev: np.ndarray = None
    cent_has: np.ndarray = None
    cent_prev_jd: np.ndarray = None
    # Domain 2
    solar_sum: np.ndarray = None
    nfr: int = 0
    sx: np.ndarray = None
    sxx: np.ndarray = None
    sy: np.ndarray = None
    sxy: np.ndarray = None
    pcos_sum: np.ndarray = None
    bspeed_sum: np.ndarray = None
    wsum_body: np.ndarray = None
    prev_dirs: np.ndarray = None
    prev_dirs_jd: float = 0.0
    prev_dirs_has: int = 0
    # Domain 5 (graph)
    cooc_sym: np.ndarray = None
    cooc_anti: np.ndarray = None

    @classmethod
    def new(cls, K, N):
        a = cls(K=K, N=N)
        z = lambda *s: np.zeros(s, dtype=np.float64)  # noqa: E731
        a.lat_sum, a.lat_sq, a.abslat_sum = z(K), z(K), z(K)
        a.node_count = np.zeros(K, dtype=np.int64)
        a.frames_present = np.zeros(K, dtype=np.int64)
        a.comp_sum, a.largest_frac_sum = z(K), z(K)
        a.drift_speed_sum, a.drift_lon_sum, a.drift_shear_sum = z(K), z(K), z(K)
        a.drift_n = np.zeros(K, dtype=np.int64)
        a.cent_prev = z(K, 3)
        a.cent_has = np.zeros(K, dtype=bool)
        a.cent_prev_jd = z(K)
        a.solar_sum = z(K)
        a.sx = z(IN_FEATS)
        a.sxx = z(IN_FEATS, IN_FEATS)
        a.sy = z(K)
        a.sxy = z(IN_FEATS, K)
        a.pcos_sum = z(K, N_BODIES, N_BODIES)
        a.bspeed_sum = z(K, N_BODIES)
        a.wsum_body = z(K)
        a.prev_dirs = z(N_BODIES, 3)
        a.cooc_sym = np.zeros((K, K), dtype=np.int64)
        a.cooc_anti = np.zeros((K, K), dtype=np.int64)
        return a

    def save(self, path: Path) -> None:
        arrs = {k: v for k, v in self.__dict__.items()
                if isinstance(v, np.ndarray)}
        scal = {k: v for k, v in self.__dict__.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)}
        import io
        buf = io.BytesIO()
        np.savez(buf, **arrs, _scalars=np.array([repr(scal)], dtype=object))
        atomic_write_bytes(path, buf.getvalue())

    @classmethod
    def load(cls, path: Path):
        import ast
        d = np.load(path, allow_pickle=True)
        a = cls(K=0, N=0)
        for k in d.files:
            if k == "_scalars":
                for sk, sv in ast.literal_eval(str(d[k][0])).items():
                    setattr(a, sk, sv)
            else:
                setattr(a, k, d[k])
        a.K, a.N = int(a.lat_sum.shape[0]), int(a.N)
        return a


def _pairwise_body_cos(dirs: np.ndarray) -> np.ndarray:
    """(10,10) cosine between geocentric body directions (unit vectors)."""
    u = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12)
    return u @ u.T


def build_antipode_map(xyz: np.ndarray) -> np.ndarray:
    """For each node, the index of the node nearest to its geographic antipode."""
    n = xyz.shape[0]
    out = np.empty(n, dtype=np.int64)
    chunk = 2048
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        # nearest to antipode(-x) == smallest dot with x
        dots = xyz[s:e] @ xyz.T
        out[s:e] = dots.argmin(axis=1)
    return out


def _flush_parquet(path: Path, columns: dict) -> None:
    """Atomically write one Parquet file (temp in same dir + os.replace)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.table({k: pa.array(v) for k, v in columns.items()})
    import io
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="snappy")
    atomic_write_bytes(path, buf.getvalue())


def run_phase2(cfg, model, model_cfg, grid, neighbors, device, state, logger):
    """Execute the adaptive temporal sweep; returns the finalized profiles + cooc."""
    K = model_cfg.codebook_size
    N = grid.n_nodes
    batch = auto_node_batch(N, cfg.node_batch)
    lat_deg = np.rad2deg(grid.lat)
    xyz = grid.xyz
    antipode = build_antipode_map(xyz)
    logger.info(f"[P2] antipode map built for {N} nodes; frames/forward={batch}")

    pq_act = cfg.parquet_dir / "activations"
    pq_day = cfg.parquet_dir / "daily_volume"
    pq_act.mkdir(parents=True, exist_ok=True)
    pq_day.mkdir(parents=True, exist_ok=True)
    acc_path = cfg.root / "accum.npz"

    # -- resume ---------------------------------------------------------------
    resume_frames = 0
    if acc_path.exists() and state.last_chunk() >= 0:
        acc = Accum.load(acc_path)
        resume_frames = acc.frame_ord
        logger.info(f"[P2] RESUME: loaded accumulators at frame_ord={resume_frames} "
                    f"(last chunk {state.last_chunk()})")
    else:
        acc = Accum.new(K, N)

    # -- adaptive clock with loud anomaly logging -----------------------------
    def on_down(jd, v):
        y = jd_to_gregorian(jd)[0]
        logger.info(f"[P2][ADAPTIVE] downshift -> 24s @ year {y} jd={jd:.4f} "
                    f"velocity={v:.4f} > thr={cfg.velocity_threshold}")

    def on_up(jd, v, nfine):
        logger.info(f"[P2][ADAPTIVE] stabilized @ jd={jd:.4f} after {nfine} micro-frames")

    clock = AdaptiveClock(cfg.start_jd, cfg.end_jd, cfg.coarse_step_seconds,
                          cfg.fine_step_seconds, cfg.velocity_threshold,
                          cfg.max_fine_run, on_down, on_up)

    # chunk-local parquet buffers
    act_frame, act_jd, act_node, act_tok = [], [], [], []
    day_didx, day_tok, day_vol = [], [], []
    chunk_id = state.last_chunk() + 1
    processed_in_chunk = 0
    t_chunk = time.time()
    t_start = time.time()
    # inner-loop heartbeat state (log every HB_FRAMES frames or HB_SECONDS seconds)
    hb_frames_mark = acc.frame_ord
    hb_time_mark = t_start
    hb_every_frames = max(1, int(cfg.heartbeat_frames))
    hb_every_seconds = float(cfg.heartbeat_seconds)

    def heartbeat(tick):
        """Granular live telemetry inside the sweep (no math changes)."""
        nonlocal hb_frames_mark, hb_time_mark
        now = time.time()
        d_frames = acc.frame_ord - hb_frames_mark
        d_time = now - hb_time_mark
        if d_frames < hb_every_frames and d_time < hb_every_seconds:
            return
        speed = d_frames / d_time if d_time > 0 else 0.0
        clk = "FINE(24s)" if tick.fine else "COARSE(1h)"
        y = jd_to_gregorian(tick.jd)[0]
        logger.info(f"[P2][HB] {format_jd(tick.jd)} (yr {y}) | "
                    f"chunk {chunk_id} frame {processed_in_chunk}/{cfg.chunk_frames} | "
                    f"{clk} | downshifts={clock.stats['downshift_events']} | "
                    f"{speed:.2f} frames/s | {format_hw(hardware_snapshot())}")
        hb_frames_mark = acc.frame_ord
        hb_time_mark = now

    def flush_chunk(cid):
        if act_frame:
            _flush_parquet(pq_act / f"chunk_{cid:06d}.parquet", {
                "frame_ord": np.asarray(act_frame, dtype=np.int64),
                "jd": np.asarray(act_jd, dtype=np.float64),
                "node_id": np.asarray(act_node, dtype=np.int32),
                "token_id": np.asarray(act_tok, dtype=np.int32)})
        if day_didx:
            _flush_parquet(pq_day / f"chunk_{cid:06d}.parquet", {
                "day_index": np.asarray(day_didx, dtype=np.int64),
                "token_id": np.asarray(day_tok, dtype=np.int32),
                "volume": np.asarray(day_vol, dtype=np.int64)})
        acc.save(acc_path)
        state.mark_chunk(cid, frame_ord=int(acc.frame_ord), n_frames=int(acc.n_frames))

    def process_frame(tokens, jd, fields_row):
        counts = np.bincount(tokens, minlength=K).astype(np.int64)
        present = np.nonzero(counts)[0]
        coverage = counts / N
        acc.frames_present[present] += 1
        acc.n_frames += 1

        # Domain 3: latitudinal / polar affinity
        np.add.at(acc.lat_sum, tokens, lat_deg)
        np.add.at(acc.lat_sq, tokens, lat_deg * lat_deg)
        np.add.at(acc.abslat_sum, tokens, np.abs(lat_deg))
        acc.node_count += counts

        # Domain 2: solar alignment angle (great-circle to subsolar point)
        s_lat, s_lon = subsolar_point(jd)
        gdist = great_circle_deg(grid.lat, grid.lon, s_lat, s_lon)
        np.add.at(acc.solar_sum, tokens, gdist)

        # Domain 3: spatial coherence & dispersion (connected components)
        n_comp, largest = connected_components(tokens, neighbors)
        for t in present:
            acc.comp_sum[t] += n_comp[int(t)]
            acc.largest_frac_sum[t] += largest[int(t)] / counts[t]

        # Domain 3: geographic drift velocity (moving centroid of each token)
        sum_xyz = np.zeros((K, 3))
        np.add.at(sum_xyz, tokens, xyz)
        for t in present:
            c = sum_xyz[t]
            nrm = np.linalg.norm(c)
            if nrm < 1e-9:
                continue
            c = c / nrm
            if acc.cent_has[t]:
                dt = jd - acc.cent_prev_jd[t]
                if dt > 0:
                    la0, lo0 = latlon_of(acc.cent_prev[t])
                    la1, lo1 = latlon_of(c)
                    ang = great_circle_deg(np.array([la0]), np.array([lo0]),
                                           la1, lo1)[0]
                    acc.drift_speed_sum[t] += ang / dt
                    dlon = (np.rad2deg(lo1 - lo0) + 180) % 360 - 180
                    acc.drift_lon_sum[t] += abs(dlon) / dt
                    acc.drift_shear_sum[t] += abs(np.rad2deg(la1 - la0)) / dt
                    acc.drift_n[t] += 1
            acc.cent_prev[t] = c
            acc.cent_prev_jd[t] = jd
            acc.cent_has[t] = True

        # Domain 2: attribution normal equations (feature node-mean vs coverage)
        fmean = fields_row.mean(axis=0)                     # (50,)
        acc.sx += fmean
        acc.sxx += np.outer(fmean, fmean)
        acc.sy += coverage
        acc.sxy += np.outer(fmean, coverage)
        acc.nfr += 1

        # Domain 2: body directions -> pairwise cos + per-body angular speed
        from ..ephemeris import global_state
        g = global_state.global_state_frame(jd)
        dirs = np.asarray(g[:, :3], dtype=np.float64)
        pcos = _pairwise_body_cos(dirs)
        acc.pcos_sum[present] += coverage[present, None, None] * pcos[None]
        acc.wsum_body[present] += coverage[present]
        if acc.prev_dirs_has and jd > acc.prev_dirs_jd:
            dt = jd - acc.prev_dirs_jd
            ud = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12)
            up = acc.prev_dirs / (np.linalg.norm(acc.prev_dirs, axis=1, keepdims=True) + 1e-12)
            cosb = np.clip((ud * up).sum(axis=1), -1.0, 1.0)
            bspeed = np.rad2deg(np.arccos(cosb)) / dt      # deg/day per body
            acc.bspeed_sum[present] += coverage[present, None] * bspeed[None]
        acc.prev_dirs = dirs
        acc.prev_dirs_jd = float(jd)
        acc.prev_dirs_has = 1

        # Domain 5 (graph): adjacency-halo symbiosis + antipodal co-occurrence
        for j in range(neighbors.shape[1]):
            nb = neighbors[:, j]
            diff = tokens != tokens[nb]
            np.add.at(acc.cooc_sym, (tokens[diff], tokens[nb][diff]), 1)
        np.add.at(acc.cooc_anti, (tokens, tokens[antipode]), 1)

        # raw activation records -> parquet buffers
        fo = acc.frame_ord
        act_frame.extend([fo] * N)
        act_jd.extend([jd] * N)
        act_node.extend(range(N))
        act_tok.extend(tokens.tolist())

        # one-frame-per-day volume series (harmonic / epoch / exclusion source)
        day_index = int(np.floor(jd - C.KALI_YUGA_EPOCH_JD))
        if day_index != acc.last_day:
            acc.last_day = day_index
            for t in present:
                day_didx.append(day_index)
                day_tok.append(int(t))
                day_vol.append(int(counts[t]))

        acc.frame_ord += 1

    # -- main loop ------------------------------------------------------------
    ticks = iter(clock)
    skipped = 0
    pend_ticks = []
    for tick in ticks:
        if skipped < resume_frames:              # fast-forward past completed work
            skipped += 1
            continue
        pend_ticks.append(tick)
        if len(pend_ticks) < batch:
            continue
        last_tick = pend_ticks[-1]
        _run_batch(pend_ticks, grid, model, device, process_frame, logger)
        pend_ticks = []
        processed_in_chunk += batch
        heartbeat(last_tick)                      # inner-loop live telemetry
        if processed_in_chunk >= cfg.chunk_frames:
            flush_chunk(chunk_id)
            speed = processed_in_chunk / max(time.time() - t_chunk, 1e-6)
            y = jd_to_gregorian(tick.jd)[0]
            logger.info(f"[P2] chunk {chunk_id} flushed @ year {y} | "
                        f"{processed_in_chunk} frames | {speed:.1f} frames/s | "
                        f"clock {clock.stats}")
            for buf in (act_frame, act_jd, act_node, act_tok,
                        day_didx, day_tok, day_vol):
                buf.clear()
            chunk_id += 1
            processed_in_chunk = 0
            t_chunk = time.time()
    if pend_ticks:
        _run_batch(pend_ticks, grid, model, device, process_frame, logger)
        processed_in_chunk += len(pend_ticks)
    if processed_in_chunk > 0 or act_frame:
        flush_chunk(chunk_id)
        logger.info(f"[P2] final chunk {chunk_id} flushed | {processed_in_chunk} frames")

    logger.info(f"[P2] sweep complete: {acc.n_frames} frames in "
                f"{time.time() - t_start:.1f}s | {clock.stats}")
    profiles = finalize_phase2(acc, logger)
    return profiles, acc


def _run_batch(ticks, grid, model, device, process_frame, logger=None):
    """Project + tokenize a batch of ticks, then accumulate each frame in order.

    Robust to rare tensor anomalies (PRD page 6): a frame whose field or tokens
    contain non-finite values, or that raises during accumulation, is logged with
    its timestamp, nullified, and skipped so the sweep never halts.
    """
    jds = [t.jd for t in ticks]
    try:
        fields = project_fields(grid, jds)                  # (B, N, 50)
        tokens_b, _ = tokenize_batch(model, fields, device, want_latent_norm=False)
    except Exception as exc:  # noqa: BLE001
        if logger:
            logger.error(f"[P2][ANOMALY] batch @ jd={jds[0]:.4f} inference failed: "
                         f"{exc!r}; nullifying {len(ticks)} frames and continuing.")
        return
    for i, tk in enumerate(ticks):
        try:
            if not np.isfinite(fields[i]).all():
                raise FloatingPointError("non-finite input field")
            process_frame(tokens_b[i], float(tk.jd), fields[i])
        except Exception as exc:  # noqa: BLE001
            if logger:
                logger.error(f"[P2][ANOMALY] frame jd={float(tk.jd):.4f} "
                             f"(res={tk.resolution_s}s) nullified: {exc!r}")


def finalize_phase2(acc: Accum, logger) -> dict:
    """Turn streaming accumulators into per-token Domain-2/Domain-3 profiles."""
    K = acc.K
    nc = acc.node_count.astype(np.float64)
    seen = nc > 0
    lat_mean = np.zeros(K)
    lat_std = np.zeros(K)
    polar = np.zeros(K)
    solar = np.zeros(K)
    lat_mean[seen] = acc.lat_sum[seen] / nc[seen]
    var = acc.lat_sq[seen] / nc[seen] - lat_mean[seen] ** 2
    lat_std[seen] = np.sqrt(np.clip(var, 0.0, None))
    polar[seen] = acc.abslat_sum[seen] / nc[seen]
    solar[seen] = acc.solar_sum[seen] / nc[seen]

    fp = acc.frames_present.astype(np.float64)
    fpos = fp > 0
    mean_comp = np.zeros(K)
    mean_largest = np.zeros(K)
    mean_comp[fpos] = acc.comp_sum[fpos] / fp[fpos]
    mean_largest[fpos] = acc.largest_frac_sum[fpos] / fp[fpos]
    dispersion = 1.0 - mean_largest                          # 0 monolith .. 1 fragmented

    dn = acc.drift_n.astype(np.float64)
    dpos = dn > 0
    drift_speed = np.zeros(K)
    drift_lon = np.zeros(K)
    drift_shear = np.zeros(K)
    drift_speed[dpos] = acc.drift_speed_sum[dpos] / dn[dpos]
    drift_lon[dpos] = acc.drift_lon_sum[dpos] / dn[dpos]
    drift_shear[dpos] = acc.drift_shear_sum[dpos] / dn[dpos]

    # -- attribution via ridge on centered normal equations -------------------
    nfr = max(acc.nfr, 1)
    mx = acc.sx / nfr
    cxx = acc.sxx / nfr - np.outer(mx, mx)
    my = acc.sy / nfr
    cxy = acc.sxy / nfr - np.outer(mx, my)                   # (50, K)
    lam = 1e-3 * (np.trace(cxx) / IN_FEATS + 1e-9)
    beta = np.linalg.solve(cxx + lam * np.eye(IN_FEATS), cxy)   # (50, K)
    body_w = np.abs(beta).reshape(N_BODIES, BODY_FEATS, K).sum(axis=1)  # (10, K)
    body_frac = body_w / (body_w.sum(axis=0, keepdims=True) + 1e-12)    # (10, K)
    top_body = body_frac.argmax(axis=0)                                 # (K,)
    top2 = np.argsort(-body_frac, axis=0)[:2]                           # (2, K)

    wb = acc.wsum_body.copy()
    wpos = wb > 0
    phase_harm = np.zeros(K)
    orb_vel = np.zeros(K)
    for t in np.nonzero(wpos)[0]:
        b1, b2 = int(top2[0, t]), int(top2[1, t])
        phase_harm[t] = acc.pcos_sum[t, b1, b2] / wb[t]
        mean_bspeed = acc.bspeed_sum[t] / wb[t]                          # (10,)
        orb_vel[t] = float((body_frac[:, t] * mean_bspeed).sum())

    profiles = {}
    for t in range(K):
        profiles[t] = {
            "lat_mean_deg": round(float(lat_mean[t]), 3),
            "lat_std_deg": round(float(lat_std[t]), 3),
            "polar_affinity_deg": round(float(polar[t]), 3),
            "mean_components": round(float(mean_comp[t]), 3),
            "dispersion_index": round(float(dispersion[t]), 4),
            "drift_speed_deg_per_day": round(float(drift_speed[t]), 5),
            "drift_longitudinal": round(float(drift_lon[t]), 5),
            "drift_shear": round(float(drift_shear[t]), 5),
            "attribution_top_body": int(top_body[t]),
            "attribution": [round(float(x), 4) for x in body_frac[:, t]],
            "phase_harmonic": round(float(phase_harm[t]), 4),
            "orbital_velocity_index": round(float(orb_vel[t]), 5),
            "solar_alignment_deg": round(float(solar[t]), 3),
            "frames_present": int(acc.frames_present[t]),
            "node_activations": int(acc.node_count[t]),
        }
    if logger:
        active = int((acc.frames_present > 0).sum())
        logger.info(f"[P2] finalized Domain-2/3 profiles: {active}/{K} tokens active; "
                    f"ridge lambda={lam:.2e}")
    return profiles
