"""The Great Indexer: unit checks + a tiny end-to-end pipeline run.

Builds a small v3 checkpoint, sweeps a short window, and asserts the master
SQLite dossier is well-formed with sane value ranges across all five domains.
Also covers the adaptive clock, connected-component labelling, atomic writes /
state lock, and resume idempotency.
"""

import sqlite3
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pyarrow")
pytest.importorskip("duckdb")

from kalachakra.ephemeris import global_state                        # noqa: E402
from kalachakra.ephemeris.calendar import parse_datetime            # noqa: E402
from kalachakra.grid import geodesic                                 # noqa: E402
from kalachakra.indexer.adaptive import AdaptiveClock               # noqa: E402
from kalachakra.indexer.config import IndexerConfig                 # noqa: E402
from kalachakra.indexer.state import StateLock, atomic_write_text   # noqa: E402
from kalachakra.indexer.sweep_math import connected_components      # noqa: E402
from kalachakra.models.autoencoder_v3 import (                      # noqa: E402
    VQAutoencoderV3, VQAutoencoderV3Config, build_knn,
)


def _tiny_ckpt(path, n=240):
    grid = geodesic.fibonacci_sphere(n)
    nb = build_knn(grid.xyz, 7)
    cfg = VQAutoencoderV3Config(n_nodes=n, in_features=50, hidden=16, latent=64,
                                fourier_modes=6, knn=7, n_blocks=2, codebook_size=4096,
                                node_chunk=128, vq_chunk=4096)
    model = VQAutoencoderV3(cfg, nb)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": "kalachakra-vqmodel-v3", "projection_version": 2,
        "config": asdict(cfg), "neighbors": np.asarray(nb, dtype=np.int64),
        "grid_xyz": np.asarray(grid.xyz, dtype=np.float64),
        "state_dict": model.state_dict(), "step": 25,
    }, path)
    return n


# ---------------------------------------------------------------------------
# component units
# ---------------------------------------------------------------------------
def test_connected_components_known_graph():
    # path 0-1-2-3; tokens A,A,B,B -> two same-token adjacent pairs, one blob each
    neighbors = np.array([[1], [2], [3], [2]])      # edges 0-1, 1-2, 2-3, 3-2
    tokens = np.array([5, 5, 9, 9])
    n_comp, largest = connected_components(tokens, neighbors)
    assert n_comp[5] == 1 and largest[5] == 2       # nodes 0,1 same token & adjacent
    assert n_comp[9] == 1 and largest[9] == 2       # nodes 2,3 same token & adjacent


def test_adaptive_clock_both_resolutions():
    if not global_state.ephemeris_available():
        pytest.skip("pyswisseph not installed")
    global_state.auto_configure()
    j0 = parse_datetime("2024-01-01T00:00:00Z")
    clk = AdaptiveClock(j0, j0 + 2.0, coarse_s=3600, fine_s=24, threshold=0.005)
    ticks = list(clk)
    res = {round(t.resolution_s) for t in ticks}
    assert 3600 in res
    # a low threshold should trigger at least one 24 s micro-frame over 2 days
    assert clk.stats["fine_ticks"] >= 1 and 24 in res


def test_atomic_write_and_state_lock(tmp_path):
    p = tmp_path / "state.json"
    atomic_write_text(p, "hello")
    assert p.read_text() == "hello"
    st = StateLock(tmp_path / "s.json")
    assert not st.phase_done("phase1")
    st.mark_phase("phase1", n=5)
    assert st.phase_done("phase1")
    st.mark_chunk(0, frame_ord=10)
    st.mark_chunk(1, frame_ord=20)
    assert st.last_chunk() == 1 and st.chunk_done(0)
    # reload from disk round-trips
    st2 = StateLock(tmp_path / "s.json")
    assert st2.phase_done("phase1") and st2.last_chunk() == 1


def _run_small_pipeline(tmp_path, days=8.0):
    from kalachakra.indexer.pipeline import run_pipeline
    ckpt = tmp_path / "checkpoints/v3/model_step_000025.pt"
    _tiny_ckpt(ckpt)
    cfg = IndexerConfig(
        checkpoint=str(ckpt), out_dir=str(tmp_path / "out"),
        start_jd=parse_datetime("2024-03-01T00:00:00Z"),
        end_jd=parse_datetime("2024-03-01T00:00:00Z") + days,
        coarse_step_seconds=10800.0, velocity_threshold=0.02,
        chunk_frames=40, calib_days=6, fft_min_samples=6, device="cpu")
    return run_pipeline(cfg), cfg


def test_pipeline_end_to_end(tmp_path):
    if not global_state.ephemeris_available():
        pytest.skip("pyswisseph not installed")
    db_path, cfg = _run_small_pipeline(tmp_path)
    assert Path(db_path).exists()

    con = sqlite3.connect(db_path)
    c = con.cursor()
    tables = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"tokens", "attribution", "transitions", "exclusion",
            "symbiosis", "antipode", "run_meta"} <= tables

    assert c.execute("SELECT COUNT(*) FROM tokens").fetchone()[0] == 4096
    # all four Domain-1 physics columns present and populated for every token
    cols = {r[1] for r in c.execute("PRAGMA table_info(tokens)")}
    for dom in ("magnitude", "dim_variance", "anomaly_isolation_percentile",
                "pc_dominant", "lat_mean_deg", "dispersion_index",
                "attribution_top_body", "solar_alignment_deg", "phase_harmonic",
                "orbital_velocity_index", "dwell_frames_mean", "fano_factor",
                "transition_top_to", "symbiosis_top_token", "antipode_top_token",
                "exclusion_top_token"):
        assert dom in cols, dom

    # value-range sanity
    ex = [r[0] for r in c.execute("SELECT corr FROM exclusion")]
    assert all(-1.0001 <= x <= 1.0001 for x in ex)
    tp = [r[0] for r in c.execute(
        "SELECT transition_top_prob FROM tokens WHERE transition_top_prob IS NOT NULL")]
    assert tp and all(0.0 <= x <= 1.0001 for x in tp)
    lat = [r[0] for r in c.execute(
        "SELECT lat_mean_deg FROM tokens WHERE node_activations>0")]
    assert lat and all(-90.001 <= x <= 90.001 for x in lat)
    pc = [r[0] for r in c.execute("SELECT DISTINCT pc_dominant FROM tokens")]
    assert set(pc) <= {1, 2, 3}
    # attribution: 10 body weights per token, summing ~1 for active tokens
    import json
    aj = c.execute("SELECT attribution_json FROM tokens WHERE node_activations>0 "
                   "LIMIT 1").fetchone()[0]
    vec = json.loads(aj)
    assert len(vec) == 10 and abs(sum(vec) - 1.0) < 1e-3
    # meta records 18 profiles / 5 domains
    prof = c.execute("SELECT value FROM run_meta WHERE key='profiles'").fetchone()[0]
    assert prof == "18"
    con.close()


def test_pipeline_resume_idempotent(tmp_path):
    if not global_state.ephemeris_available():
        pytest.skip("pyswisseph not installed")
    db1, cfg = _run_small_pipeline(tmp_path)
    mtime1 = Path(db1).stat().st_mtime
    # second run: all phases already marked done -> loads from json, rewrites DB
    from kalachakra.indexer.pipeline import run_pipeline
    st = StateLock(cfg.state_path)
    assert all(st.phase_done(p) for p in ("phase1", "phase2", "phase3", "phase4"))
    db2 = run_pipeline(cfg)
    assert Path(db2) == Path(db1)
    assert Path(db2).stat().st_mtime >= mtime1
