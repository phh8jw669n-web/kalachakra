"""Tests for the Phase-3 persistence layer: H3, mipmap, Parquet, DuckDB."""

import numpy as np
import pytest

from kalachakra.geo import h3index
from kalachakra.storage import mipmap


# --- H3 indexing ----------------------------------------------------------
@pytest.mark.skipif(not h3index.h3_available(), reason="h3 not installed")
def test_h3_cell_is_stable_and_hierarchical():
    a = h3index.cell_for(27.0, 78.0, 4)
    b = h3index.cell_for(27.0, 78.0, 4)
    assert a == b and a > 0
    fine = h3index.cell_for(27.0, 78.0, 7)
    assert h3index.parent(fine, 4) == a


@pytest.mark.skipif(not h3index.h3_available(), reason="h3 not installed")
def test_h3_bbox_contains_interior_point():
    cells = set(h3index.cells_in_bbox(20, 70, 30, 80, 4))
    assert h3index.cell_for(25.0, 75.0, 4) in cells


def test_h3_grid_cells_shape():
    lat = np.array([0.0, 45.0, -45.0])
    lng = np.array([0.0, 90.0, -90.0])
    cells = h3index.cells_for_grid(lat, lng, 4)
    assert cells.shape == (3,) and cells.dtype == np.int64


# --- temporal mipmapping --------------------------------------------------
def test_bucket_reductions_ragged():
    v = np.arange(10, dtype=np.float64)  # 3 buckets of 4,4,2
    assert np.array_equal(mipmap.bucket_max(v, 4), [3, 7, 9])
    assert np.allclose(mipmap.bucket_mean(v, 4), [1.5, 5.5, 8.5])


def test_mode_per_bucket_picks_dominant_token():
    tok = np.array([1, 1, 1, 2, 5, 5, 5, 5])
    m = mipmap.mode_per_bucket(tok, 4, n_tokens=8)
    assert list(m) == [1, 5]


def test_hourly_rollup_bucket_count():
    n = 300  # exactly 2 hourly buckets
    pot = np.random.default_rng(0).random(n)
    shear = np.random.default_rng(1).random(n)
    leaf = np.random.default_rng(2).integers(0, 4096, n)
    roll = mipmap.hourly_rollup(pot, shear, leaf)
    assert roll["max_potential"].shape == (2,)
    assert roll["archetype"].shape == (2,)


def test_select_tier_thresholds():
    assert mipmap.select_tier(500) == "tier1"
    assert mipmap.select_tier(50_000) == "tier2"       # /150 <= 1000
    assert mipmap.select_tier(500_000_000) == "tier3"


# --- Parquet + DuckDB round trip -----------------------------------------
pa = pytest.importorskip("pyarrow")
duck = pytest.importorskip("duckdb")


def _sample_columns(n=200, jd0=2451545.0):
    from kalachakra.ephemeris import timeline
    from kalachakra.grid import geodesic
    grid = geodesic.fibonacci_sphere(8)
    rng = np.random.default_rng(0)
    frames = np.arange(n)
    jd = jd0 + frames * (24 / 86400)
    node = rng.integers(0, 8, n)
    lat = np.rad2deg(grid.lat[node]); lng = np.rad2deg(grid.lon[node])
    macro = rng.integers(0, 64, n); micro = rng.integers(0, 64, n)
    return {
        "jd": jd.astype(np.float64),
        "frame": frames.astype(np.int64),
        "node": node.astype(np.int32),
        "lat": lat.astype(np.float32), "lng": lng.astype(np.float32),
        "h3": h3index.cells_for_grid(lat, lng, 4),
        "macro": macro.astype(np.int16), "micro": micro.astype(np.int16),
        "leaf": (macro * 64 + micro).astype(np.int32),
        "rarity": rng.random(n).astype(np.float32),
        "potential": rng.random(n).astype(np.float32),
        "shear": rng.random(n).astype(np.float32),
    }


def _grid_columns(n_frames, n_nodes, jd0=2451545.0):
    """A regular (frame x node) block, reshapeable for tier-2/3 rollups."""
    from kalachakra.grid import geodesic
    grid = geodesic.fibonacci_sphere(n_nodes)
    rng = np.random.default_rng(1)
    frame = np.repeat(np.arange(n_frames), n_nodes)
    node = np.tile(np.arange(n_nodes), n_frames)
    jd = jd0 + frame * (24 / 86400)
    lat = np.rad2deg(grid.lat[node]); lng = np.rad2deg(grid.lon[node])
    macro = rng.integers(0, 64, n_frames * n_nodes)
    micro = rng.integers(0, 64, n_frames * n_nodes)
    return {
        "jd": jd.astype(np.float64), "frame": frame.astype(np.int64),
        "node": node.astype(np.int32),
        "lat": lat.astype(np.float32), "lng": lng.astype(np.float32),
        "h3": h3index.cells_for_grid(lat, lng, 4),
        "macro": macro.astype(np.int16), "micro": micro.astype(np.int16),
        "leaf": (macro * 64 + micro).astype(np.int32),
        "rarity": rng.random(n_frames * n_nodes).astype(np.float32),
        "potential": rng.random(n_frames * n_nodes).astype(np.float32),
        "shear": rng.random(n_frames * n_nodes).astype(np.float32),
    }


def _load_build_index():
    import importlib.util
    from pathlib import Path
    p = Path(__file__).resolve().parents[1] / "scripts" / "build_index.py"
    spec = importlib.util.spec_from_file_location("build_index", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_tier3_daily_rollups_written_and_routed(tmp_path):
    from kalachakra.storage.duckdb_engine import DuckDBEngine, ViewportQuery
    from kalachakra.storage.parquet_store import ParquetTokenStore

    n_nodes = 4
    n_frames = mipmap.FRAMES_PER_DAY * 2 + 10   # 2 full daily buckets + a partial
    cols = _grid_columns(n_frames, n_nodes)
    store = ParquetTokenStore(tmp_path / "index")
    store.write_frames(cols)

    bi = _load_build_index()
    bi._write_daily(store, cols, n_nodes)
    assert store.has_tier("tier3")
    assert list((tmp_path / "index" / "tier3").glob("century=*"))

    engine = DuckDBEngine(store)
    # A deep-time (epochal) span must route to tier3 and still return rows.
    q = ViewportQuery(min_lat=-90, min_lng=-180, max_lat=90, max_lng=180,
                      start_jd=cols["jd"][0], end_jd=cols["jd"][0] + 4_000_000.0,
                      rarity_min=0.0, limit=50)
    assert engine._tier_for(q) == "tier3"
    rows = engine.query(q)
    assert len(rows) > 0 and "anomaly_count" in rows[0]
    engine.close()


def test_store_stamps_and_reads_projection_version(tmp_path):
    from kalachakra import constants as C
    from kalachakra.storage.parquet_store import ParquetTokenStore

    store = ParquetTokenStore(tmp_path / "idx")
    assert store.projection_version() == 1        # legacy default when unstamped
    store.write_meta(nodes=256)
    meta = store.read_meta()
    assert meta["projection_version"] == C.PROJECTION_VERSION
    assert meta["nodes"] == 256
    assert ParquetTokenStore(tmp_path / "idx").projection_version() == C.PROJECTION_VERSION


def test_sparse_streaming_build_keeps_only_rare_rows(tmp_path):
    """Full-scale path: --rarity-min streams a two-pass build whose fine tiers
    (tier1/tier2) are rarity-thresholded while tier3 stays dense — so the index
    stays storable at scale instead of writing every frame x node."""
    pytest.importorskip("torch")
    from kalachakra.ephemeris import global_state
    if not global_state.ephemeris_available():
        pytest.skip("pyswisseph not installed")

    import torch
    bi = _load_build_index()
    out = tmp_path / "sparse"
    nodes, frames, rmin = 24, 200, 0.3
    # Seed so the fresh (untrained) model's token distribution is deterministic;
    # otherwise the fraction above a fixed rarity threshold varies run to run.
    torch.manual_seed(0); np.random.seed(0)
    rc = bi.main(["--out", str(out), "--nodes", str(nodes), "--frames", str(frames),
                  "--rarity-min", str(rmin)])
    assert rc == 0

    import glob

    import duckdb

    from kalachakra.storage.parquet_store import ParquetTokenStore
    store = ParquetTokenStore(out)
    assert store.has_tier("tier1") and store.has_tier("tier3")

    dense = frames * nodes
    t1 = glob.glob(str(store._tier_dir("tier1")) + "/**/*.parquet", recursive=True)
    n_t1, min_rar = duckdb.sql(
        f"SELECT count(*), min(rarity) FROM read_parquet({t1!r})").fetchone()
    # tier1 is sparse (far below dense frames x nodes) and every row clears the bar.
    assert 0 < n_t1 < dense
    assert min_rar >= rmin - 1e-6
    # tier3 is the dense base layer: one daily bucket per node here.
    t3 = glob.glob(str(store._tier_dir("tier3")) + "/**/*.parquet", recursive=True)
    n_t3 = duckdb.sql(f"SELECT count(*) FROM read_parquet({t3!r})").fetchone()[0]
    assert n_t3 == nodes


def test_parquet_write_and_duckdb_query(tmp_path):
    from kalachakra.storage.duckdb_engine import DuckDBEngine, ViewportQuery
    from kalachakra.storage.parquet_store import ParquetTokenStore

    store = ParquetTokenStore(tmp_path / "index")
    cols = _sample_columns(200)
    written = store.write_frames(cols)
    assert written == 200
    # partitioned dataset created
    assert list((tmp_path / "index" / "tier1").glob("century=*"))

    engine = DuckDBEngine(store)
    pmf = engine.token_pmf("tier1")
    assert sum(pmf.values()) == 200

    q = ViewportQuery(min_lat=-90, min_lng=-180, max_lat=90, max_lng=180,
                      start_jd=cols["jd"][0], end_jd=cols["jd"][-1],
                      rarity_min=0.0, limit=50)
    rows = engine.query(q)
    assert 0 < len(rows) <= 50
    # ordered by rarity descending
    rar = [r["rarity"] for r in rows]
    assert rar == sorted(rar, reverse=True)
    engine.close()
