"""Configuration for the version7 regional/city-grid engine.

Reuses :class:`version6.config.SirenConfig` verbatim (same bounded, soft-clamped L*a*b*
head) and adds the structured-dataset knobs (cities + regional grid) and the render-grid
manifest the frontend consumes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields

from version6.config import SirenConfig
from version6.ephemeris import J2000

#: +/- 5000 Julian years around J2000 — the ~10,000-year timeline.
_HALF_SPAN_DAYS = 5000.0 * 365.25
DEFAULT_JD_START = J2000 - _HALF_SPAN_DAYS
DEFAULT_JD_END = J2000 + _HALF_SPAN_DAYS


@dataclass
class GridConfig:
    """The equirectangular render grid the frontend bakes into a texture."""
    width: int = 180                  # longitude nodes (-180..180)
    height: int = 90                  # latitude nodes (-90..90)


@dataclass
class DataConfig:
    """The structured sampler: cities + a regional lat/lon lattice, over the timeline."""
    jd_start: float = DEFAULT_JD_START
    jd_end: float = DEFAULT_JD_END
    batch: int = 2048
    #: fraction of each batch drawn from (jittered) curated cities …
    city_frac: float = 0.35
    #: … from the regional lat/lon grid nodes (jittered) …
    grid_frac: float = 0.45
    #: … and the remainder drawn uniformly, so the field stays globally valid between nodes.
    #: (city_frac + grid_frac + uniform = 1; uniform is whatever is left over.)
    grid_step_deg: float = 5.0        # regional lattice spacing
    jitter_deg: float = 2.0           # spatial jitter around city/grid nodes (smooth fields)
    seed: int = 0


@dataclass
class TrainConfig:
    lr: float = 1e-4
    lr_min: float = 1e-6
    weight_decay: float = 0.0
    warmup_steps: int = 500
    max_steps: int = 40_000
    grad_clip: float = 1.0
    color_scale: float = 20.0         # target ||dLab|| = color_scale * ||dSky||
    anchor_weight: float = 0.05       # gauge anchor toward neutral L*=60
    device: str = ""
    num_workers: int = 0
    out_dir: str = "version7/checkpoints"
    save_every: int = 2000
    log_every: int = 25
    seed: int = 0


def _only_known(dc, d: dict) -> dict:
    known = {f.name for f in fields(dc)}
    return {k: v for k, v in d.items() if k in known}


@dataclass
class V7Config:
    siren: SirenConfig = field(default_factory=SirenConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "V7Config":
        return cls(
            siren=SirenConfig(**_only_known(SirenConfig, d.get("siren", {}))),
            grid=GridConfig(**_only_known(GridConfig, d.get("grid", {}))),
            data=DataConfig(**_only_known(DataConfig, d.get("data", {}))),
            train=TrainConfig(**_only_known(TrainConfig, d.get("train", {}))),
        )
