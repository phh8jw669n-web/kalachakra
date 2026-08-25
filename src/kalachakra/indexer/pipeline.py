"""Orchestrator for the Great Indexer: runs the phases, resumable and crash-safe.

Phase order: 1 (tensor physics, in-GPU) -> 2 (adaptive temporal sweep, flush to
Parquet) -> 3 (DuckDB temporal waveforms) -> 4 (DuckDB + graph ecosystem) ->
master SQLite DB. Each phase's output is persisted atomically so a restart skips
completed phases; the sweep additionally resumes at the exact frame via the
accumulator checkpoint.
"""

from __future__ import annotations

import json
import time

import numpy as np

from ..ephemeris import global_state
from ..ephemeris.calendar import format_jd
from .config import IndexerConfig
from .master_db import write_master
from .model_io import auto_node_batch, load_model_and_grid, select_device
from .phase1_physics import run_phase1
from .phase2_sweep import Accum, run_phase2
from .phase3_temporal import run_phase3
from .phase4_ecosystem import empty_phase4, run_phase4
from .state import StateLock, atomic_write_text
from .telemetry import DiskWriteRate, format_hw, hardware_snapshot, setup_logging


def _save_json(path, obj):
    atomic_write_text(path, json.dumps(obj))


def _load_json(path):
    return json.loads(path.read_text())


def run_pipeline(cfg: IndexerConfig):
    cfg.root.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(cfg.log_dir)
    disk = DiskWriteRate()
    t0 = time.time()
    logger.info("=" * 78)
    logger.info("GREAT INDEXER starting")
    logger.info(f"window: {format_jd(cfg.start_jd)}  ->  {format_jd(cfg.end_jd)} "
                f"({(cfg.end_jd - cfg.start_jd) / 365.25:.2f} yr)")
    logger.info(f"adaptive: coarse={cfg.coarse_step_seconds}s fine={cfg.fine_step_seconds}s "
                f"threshold={cfg.velocity_threshold}")
    if cfg.lite:
        logger.info("MODE: --lite (skipping Domain-5 ecosystem graph + Domain-1 PCA; "
                    "Domains 1-4 kept; skipped columns written as NULL)")
    logger.info(f"hardware: {format_hw(hardware_snapshot(disk))}")

    if not global_state.ephemeris_available():
        raise RuntimeError("pyswisseph unavailable; cannot run the sweep.")
    global_state.auto_configure()

    device = select_device(cfg.device)
    model, model_cfg, grid, neighbors = load_model_and_grid(cfg.checkpoint, device)
    cfg.codebook_size = int(model_cfg.codebook_size)
    cfg.n_nodes = int(grid.n_nodes)
    logger.info(f"model on {device}: {cfg.n_nodes:,} nodes, "
                f"codebook={cfg.codebook_size}, latent={model_cfg.latent}")

    state = StateLock(cfg.state_path)
    state.set_config(cfg.to_dict())

    p1_path = cfg.root / "phase1.json"
    p2_path = cfg.root / "phase2.json"
    p3_path = cfg.root / "phase3.json"
    p4_path = cfg.root / "phase4.json"
    rel_path = cfg.root / "relations.json"

    # -- Phase 1: tensor physics ---------------------------------------------
    if state.phase_done("phase1") and p1_path.exists():
        logger.info("[P1] already complete; loading.")
        phase1 = {int(k): v for k, v in _load_json(p1_path).items()}
    else:
        logger.info("[P1] Domain-1 tensor physics ...")
        calib = np.linspace(cfg.start_jd, cfg.end_jd, cfg.calib_days, endpoint=False)
        batch = auto_node_batch(cfg.n_nodes, cfg.node_batch)
        phase1 = run_phase1(model, grid, device, calib.tolist(), batch, logger,
                            lite=cfg.lite)
        _save_json(p1_path, {str(k): v for k, v in phase1.items()})
        state.mark_phase("phase1", n_tokens=len(phase1))
        logger.info(f"[P1] done. {format_hw(hardware_snapshot(disk))}")

    # -- Phase 2: adaptive sweep ---------------------------------------------
    if state.phase_done("phase2") and p2_path.exists():
        logger.info("[P2] already complete; loading profiles + accumulators.")
        phase2 = {int(k): v for k, v in _load_json(p2_path).items()}
        acc = Accum.load(cfg.root / "accum.npz")
    else:
        logger.info("[P2] Domain-2/3 adaptive temporal sweep ...")
        phase2, acc = run_phase2(cfg, model, model_cfg, grid, neighbors, device,
                                 state, logger)
        _save_json(p2_path, {str(k): v for k, v in phase2.items()})
        state.mark_phase("phase2", n_tokens=len(phase2))
        logger.info(f"[P2] done. {format_hw(hardware_snapshot(disk))}")

    # -- Phase 3: temporal waveforms (DuckDB) --------------------------------
    if state.phase_done("phase3") and p3_path.exists():
        logger.info("[P3] already complete; loading.")
        phase3 = {int(k): v for k, v in _load_json(p3_path).items()}
    else:
        logger.info("[P3] Domain-4 temporal waveforms (DuckDB) ...")
        phase3 = run_phase3(cfg, logger)
        _save_json(p3_path, {str(k): v for k, v in phase3.items()})
        state.mark_phase("phase3", n_tokens=len(phase3))
        logger.info(f"[P3] done. {format_hw(hardware_snapshot(disk))}")

    # -- Phase 4: ecosystem (DuckDB + graph) ---------------------------------
    if state.phase_done("phase4") and p4_path.exists() and rel_path.exists():
        logger.info("[P4] already complete; loading.")
        phase4 = {int(k): v for k, v in _load_json(p4_path).items()}
        relations = _load_json(rel_path)
    elif cfg.lite:
        logger.info("[P4] Domain-5 ecosystem SKIPPED (--lite); "
                    "writing NULL profile columns + empty relation graphs.")
        phase4, relations = empty_phase4(cfg.codebook_size)
        _save_json(p4_path, {str(k): v for k, v in phase4.items()})
        _save_json(rel_path, relations)
        state.mark_phase("phase4", n_tokens=len(phase4), lite=True)
    else:
        logger.info("[P4] Domain-5 ecosystem (DuckDB + mesh graph) ...")
        phase4, relations = run_phase4(cfg, acc, logger)
        _save_json(p4_path, {str(k): v for k, v in phase4.items()})
        _save_json(rel_path, relations)
        state.mark_phase("phase4", n_tokens=len(phase4))
        logger.info(f"[P4] done. {format_hw(hardware_snapshot(disk))}")

    # -- Master DB ------------------------------------------------------------
    meta = {
        "start_jd": cfg.start_jd, "end_jd": cfg.end_jd,
        "start": format_jd(cfg.start_jd), "end": format_jd(cfg.end_jd),
        "codebook_size": cfg.codebook_size, "n_nodes": cfg.n_nodes,
        "velocity_threshold": cfg.velocity_threshold,
        "coarse_step_seconds": cfg.coarse_step_seconds,
        "fine_step_seconds": cfg.fine_step_seconds,
        "checkpoint": cfg.checkpoint,
        "elapsed_seconds": round(time.time() - t0, 2),
        "lite": cfg.lite,
        # lite drops Domain 5 (4 profiles) and Domain-1 PCA (1 profile)
        "domains": 4 if cfg.lite else 5,
        "profiles": 13 if cfg.lite else 18,
    }
    db_path = write_master(cfg, phase1, phase2, phase3, phase4, relations, meta, logger)
    state.mark_phase("master_db", path=str(db_path))
    logger.info(f"GREAT INDEXER complete in {time.time() - t0:.1f}s -> {db_path}")
    logger.info(f"final hardware: {format_hw(hardware_snapshot(disk))}")
    logger.info("=" * 78)
    return db_path
