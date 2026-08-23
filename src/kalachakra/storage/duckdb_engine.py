"""
Embedded DuckDB query router over the Parquet token datasets (blueprint §3).

An in-process, vectorized SQL engine that answers viewport queries by combining:
  * temporal partition pruning (century filter + jd range),
  * dynamic mipmap-tier selection from the requested span / playback velocity,
  * H3 cell-set spatial filtering (constant-time integer membership),
  * rarity-threshold evaluation,
bounding the scan to ~1000 rows regardless of whether the window is three days or
ten millennia.

Requires duckdb. Spatial filtering uses :mod:`kalachakra.geo.h3index`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

try:  # pragma: no cover - optional dependency
    import duckdb

    _HAS_DUCKDB = True
except Exception:  # noqa: BLE001
    duckdb = None
    _HAS_DUCKDB = False

from ..geo import h3index
from . import mipmap
from .parquet_store import ParquetTokenStore


def duckdb_available() -> bool:
    return _HAS_DUCKDB


@dataclass
class ViewportQuery:
    min_lat: float
    min_lng: float
    max_lat: float
    max_lng: float
    start_jd: float
    end_jd: float
    velocity: float = 1.0        # playback stride (frames/step); informs tier
    rarity_min: float = 0.0
    limit: int = 1000


class DuckDBEngine:
    """Routes viewport queries to the optimal tier and returns bounded results."""

    def __init__(self, store: ParquetTokenStore):
        if not _HAS_DUCKDB:
            raise RuntimeError("duckdb is required. `pip install duckdb`.")
        self.store = store
        self.con = duckdb.connect(":memory:")

    def _tier_for(self, q: ViewportQuery) -> str:
        span_frames = max(1, int((q.end_jd - q.start_jd) * 86400 / 24))
        # Fast scrubbing (large velocity) also coarsens the tier. Tier selection
        # bounds the SCAN to ~1000 rows (spec), independent of the result limit.
        effective = int(span_frames / max(q.velocity, 1.0))
        tier = mipmap.select_tier(effective, target_rows=1000)
        # Fall back to a finer tier if the coarser rollup has not been built yet.
        for t in ("tier3", "tier2", "tier1")[("tier3", "tier2", "tier1").index(tier):]:
            if self.store.has_tier(t):
                return t
        return "tier1"

    #: Above this many H3 cells the IN-list is impractical (large/global bbox);
    #: fall back to a lat/lng range predicate that scans the pruned partitions.
    MAX_CELLS = 4000

    def _spatial_predicate(self, q: ViewportQuery) -> str:
        """H3 set membership for regional boxes, lat/lng range for large ones."""
        cells = h3index.cells_in_bbox(q.min_lat, q.min_lng, q.max_lat, q.max_lng,
                                      h3index.BASE_RESOLUTION)
        if 0 < len(cells) <= self.MAX_CELLS:
            return "h3 IN (" + ",".join(str(c) for c in cells) + ")"
        return (f"lat BETWEEN {q.min_lat} AND {q.max_lat} "
                f"AND lng BETWEEN {q.min_lng} AND {q.max_lng}")

    def query(self, q: ViewportQuery) -> list[dict]:
        """Execute a viewport query; returns up to ``limit`` rows as dicts."""
        tier = self._tier_for(q)
        glob = self.store.tier_glob(tier)
        # Century pruning bounds so the planner skips irrelevant partitions.
        from .parquet_store import century_of
        c0, c1 = century_of(q.start_jd), century_of(q.end_jd)
        spatial = self._spatial_predicate(q)

        sql = f"""
            SELECT * FROM read_parquet('{glob}', hive_partitioning=1)
            WHERE century BETWEEN {c0} AND {c1}
              AND jd BETWEEN {q.start_jd} AND {q.end_jd}
              AND rarity >= {q.rarity_min}
              AND {spatial}
            ORDER BY rarity DESC
            LIMIT {q.limit}
        """
        try:
            rel = self.con.execute(sql)
            cols = [d[0] for d in rel.description]
            return [dict(zip(cols, row)) for row in rel.fetchall()]
        except duckdb.IOException:
            # No parquet files for the selected tier yet.
            return []

    def token_pmf(self, tier: str = "tier1") -> dict[int, int]:
        """Empirical leaf-token histogram over a tier (for the rarity model)."""
        glob = self.store.tier_glob(tier)
        try:
            rel = self.con.execute(
                f"SELECT leaf, COUNT(*) c FROM read_parquet('{glob}') GROUP BY leaf")
            return {int(leaf): int(c) for leaf, c in rel.fetchall()}
        except duckdb.IOException:
            return {}

    def close(self) -> None:
        self.con.close()
