"""The Great Indexer: deep-time archetype profiling pipeline.

An out-of-core pipeline that sweeps the 10,256-year ephemeris with adaptive
time-stepping, uses PyTorch purely as the physics engine, flushes activation
records to compressed Parquet, aggregates them with DuckDB, and compiles 18
mathematical profiles (five domains) for all 4096 VQ archetypes into a single
queryable SQLite dossier database.

See :mod:`kalachakra.indexer.pipeline` for the orchestrator.
"""

from .config import IndexerConfig

__all__ = ["IndexerConfig"]
