"""Configuration for the Great Indexer (deep-time archetype profiler).

A single frozen-ish dataclass carries every knob so the pipeline is fully
reproducible from the state file. Defaults describe the full 10,256-year sweep;
tests and MVP runs shrink ``start_jd`` / ``end_jd`` and point ``checkpoint`` at a
small model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from .. import constants as C


@dataclass
class IndexerConfig:
    # -- inputs / outputs ----------------------------------------------------
    checkpoint: str = "checkpoints/v3/model_step_000025.pt"
    out_dir: str = "index_out"                      # parquet + state + master db live here

    # -- temporal window (defaults to the whole timeline) --------------------
    start_jd: float = C.KALI_YUGA_EPOCH_JD
    end_jd: float = C.KALI_YUGA_EPOCH_JD + C.TIMELINE_YEARS * C.DAYS_PER_YEAR

    # -- adaptive time-stepping ---------------------------------------------
    coarse_step_seconds: float = 3600.0             # 1 hour default cruise
    fine_step_seconds: float = float(C.VIGHATIKA_SECONDS)   # 24 s micro-resolution
    #: downshift when ||dG/dt|| per hour exceeds this (rad-equivalent of the
    #: stacked body direction tensor). Calibrated to fire on lunar transits /
    #: fast aspect formation, not on stable planetary cruise.
    velocity_threshold: float = 0.02
    #: never emit two fine windows without at least this cool-down (frames).
    max_fine_run: int = 20_000                      # safety cap per event

    # -- chunking / flushing / checkpointing --------------------------------
    chunk_frames: int = 50_000                      # processed frames per parquet flush + state lock
    node_batch: int = 0                             # 0 -> auto from node count
    calib_days: int = 24                            # frames for the pre-quant magnitude calibration
    # inner-loop heartbeat: log live telemetry every N frames OR every S seconds
    heartbeat_frames: int = 100
    heartbeat_seconds: float = 60.0

    # -- analytics -----------------------------------------------------------
    epoch_years: int = 50                           # Fano-factor bin width
    top_k_relations: int = 8                        # transitions / exclusions kept per token
    fft_min_samples: int = 16                       # skip harmonic FFT below this many daily points

    # -- device --------------------------------------------------------------
    device: str = ""                                # "" -> auto (mps/cuda/cpu)
    seed: int = 0

    # -- provenance ----------------------------------------------------------
    codebook_size: int = 0                          # filled from the checkpoint
    n_nodes: int = 0                                # filled from the checkpoint
    extra: dict = field(default_factory=dict)

    @property
    def root(self) -> Path:
        return Path(self.out_dir)

    @property
    def parquet_dir(self) -> Path:
        return self.root / "parquet"

    @property
    def state_path(self) -> Path:
        return self.root / "state.json"

    @property
    def log_dir(self) -> Path:
        return self.root / "logs"

    @property
    def master_db_path(self) -> Path:
        return self.root / "dossiers.sqlite"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "IndexerConfig":
        known = {f: d[f] for f in cls.__dataclass_fields__ if f in d}
        return cls(**known)
