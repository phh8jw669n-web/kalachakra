"""
Partitioned Apache Parquet persistence for quantized states (blueprint §3).

Tier-1 descriptors (one row per frame x node) are written to Parquet partitioned
by century, with dictionary-encoded token columns and Snappy compression so an
analytical engine reads only the columns and partitions a query touches. Tier-2
hourly rollups (from :mod:`kalachakra.storage.mipmap`) are written to a sibling
dataset.

Requires pyarrow.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:  # pragma: no cover - optional dependency
    import pyarrow as pa
    import pyarrow.parquet as pq

    _HAS_PYARROW = True
except Exception:  # noqa: BLE001
    pa = pq = None
    _HAS_PYARROW = False

from ..ephemeris.calendar import jd_to_gregorian


def pyarrow_available() -> bool:
    return _HAS_PYARROW


def century_of(jd: float) -> int:
    """Century partition key (astronomical year // 100) for a Julian Day."""
    year = jd_to_gregorian(float(jd))[0]
    return int(np.floor(year / 100.0))


def _require():
    if not _HAS_PYARROW:
        raise RuntimeError("pyarrow is required. `pip install \"kalachakra[cluster]\"` "
                           "or `pip install pyarrow`.")


# Token/state columns kept dictionary-encoded (cheap, repetitive) vs. Snappy for
# the continuous floats.
_DICT_COLUMNS = ["macro", "micro", "leaf", "archetype"]


class ParquetTokenStore:
    """Writer/locator for the tiered Parquet token datasets."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.tier1 = self.root / "tier1"
        self.tier2 = self.root / "tier2"
        self.tier3 = self.root / "tier3"

    # -- tier 1: native frames -------------------------------------------
    def write_frames(self, columns: dict[str, np.ndarray]) -> int:
        """Write a batch of per-(frame,node) rows, partitioned by century.

        ``columns`` must include at least ``jd`` and the token/metric columns.
        Returns the number of rows written.
        """
        _require()
        n = len(next(iter(columns.values())))
        century = np.array([century_of(j) for j in columns["jd"]], dtype=np.int32)
        table = _to_table({**columns, "century": century})
        pq.write_to_dataset(
            table, root_path=str(self.tier1),
            partition_cols=["century"], compression="snappy",
            use_dictionary=[c for c in _DICT_COLUMNS if c in table.column_names],
            existing_data_behavior="overwrite_or_ignore",
        )
        return n

    # -- tier 2: hourly rollups ------------------------------------------
    def write_hourly(self, columns: dict[str, np.ndarray]) -> int:
        _require()
        n = len(next(iter(columns.values())))
        century = np.array([century_of(j) for j in columns["jd"]], dtype=np.int32)
        table = _to_table({**columns, "century": century})
        pq.write_to_dataset(
            table, root_path=str(self.tier2),
            partition_cols=["century"], compression="snappy",
            use_dictionary=[c for c in _DICT_COLUMNS if c in table.column_names],
            existing_data_behavior="overwrite_or_ignore",
        )
        return n

    # -- tier 3: daily / epochal rollups ---------------------------------
    def write_daily(self, columns: dict[str, np.ndarray]) -> int:
        _require()
        n = len(next(iter(columns.values())))
        century = np.array([century_of(j) for j in columns["jd"]], dtype=np.int32)
        table = _to_table({**columns, "century": century})
        pq.write_to_dataset(
            table, root_path=str(self.tier3),
            partition_cols=["century"], compression="snappy",
            use_dictionary=[c for c in _DICT_COLUMNS if c in table.column_names],
            existing_data_behavior="overwrite_or_ignore",
        )
        return n

    def _tier_dir(self, tier: str) -> Path:
        return {"tier1": self.tier1, "tier2": self.tier2, "tier3": self.tier3}[tier]

    def has_tier(self, tier: str) -> bool:
        d = self._tier_dir(tier)
        return d.exists() and any(d.glob("**/*.parquet"))

    def tier_glob(self, tier: str) -> str:
        """Glob pattern for a tier's Parquet files (for DuckDB scans)."""
        return str(self._tier_dir(tier) / "**" / "*.parquet")


def _to_table(columns: dict[str, np.ndarray]):
    arrays, names = [], []
    for name, arr in columns.items():
        arr = np.asarray(arr)
        if arr.ndim == 1:
            arrays.append(pa.array(arr))
            names.append(name)
        elif arr.ndim == 2:
            # e.g. the 64-d latent -> fixed-size list column.
            arrays.append(pa.array(list(arr.astype(np.float32))))
            names.append(name)
        else:
            raise ValueError(f"unsupported column ndim for {name!r}: {arr.ndim}")
    return pa.table(arrays, names=names)
