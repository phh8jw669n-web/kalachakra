"""Configuration for the version8 (88-D SIREN) engine. Self-contained — no cross-version imports."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields

from .ephemeris import J2000
from .state import STATE_DIM

_HALF_SPAN_DAYS = 5000.0 * 365.25
DEFAULT_JD_START = J2000 - _HALF_SPAN_DAYS
DEFAULT_JD_END = J2000 + _HALF_SPAN_DAYS


@dataclass
class SirenConfig:
    in_features: int = STATE_DIM      # 88
    hidden: int = 128
    hidden_layers: int = 4
    out_features: int = 3
    omega0: float = 30.0
    #: gamut-bounded head: L* = l0 + lspan*sigmoid(z0); a*,b* = ab*tanh(z)
    lab_l0: float = 5.0
    lab_lspan: float = 90.0
    lab_ab: float = 80.0


@dataclass
class DataConfig:
    jd_start: float = DEFAULT_JD_START
    jd_end: float = DEFAULT_JD_END
    lat_min: float = -90.0
    lat_max: float = 90.0
    lon_min: float = -180.0
    lon_max: float = 180.0
    batch: int = 2048
    seed: int = 0


@dataclass
class TrainConfig:
    lr: float = 1e-4
    lr_min: float = 1e-6
    weight_decay: float = 0.0
    warmup_steps: int = 500
    max_steps: int = 40_000
    grad_clip: float = 1.0
    gamma: float = 15.0               # colour scale: ||dLab|| = gamma * d_sky
    anchor_weight: float = 0.05
    device: str = ""
    num_workers: int = 0
    out_dir: str = "version8/checkpoints"
    save_every: int = 2000
    log_every: int = 25
    seed: int = 0


def _only_known(dc, d: dict) -> dict:
    known = {f.name for f in fields(dc)}
    return {k: v for k, v in d.items() if k in known}


@dataclass
class V8Config:
    siren: SirenConfig = field(default_factory=SirenConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "V8Config":
        return cls(
            siren=SirenConfig(**_only_known(SirenConfig, d.get("siren", {}))),
            data=DataConfig(**_only_known(DataConfig, d.get("data", {}))),
            train=TrainConfig(**_only_known(TrainConfig, d.get("train", {}))),
        )
