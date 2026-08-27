"""Configuration dataclasses for the version6 SIREN engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields

from .ephemeris import J2000, STATE_DIM

#: Default sampling span: +/- 5000 Julian years around J2000 (~10,000-year ocean of time).
_HALF_SPAN_DAYS = 5000.0 * 365.25
DEFAULT_JD_START = J2000 - _HALF_SPAN_DAYS
DEFAULT_JD_END = J2000 + _HALF_SPAN_DAYS


@dataclass
class SirenConfig:
    in_features: int = STATE_DIM      # 33
    hidden: int = 48
    hidden_layers: int = 2
    out_features: int = 3             # CIE L*a*b*
    omega0: float = 30.0
    #: Bounded output gauge. The linear head's raw logits are squashed so the colour is
    #: ALWAYS displayable — L* in (0,100), a*/b* in (-lab_ab, lab_ab) — via a tanh that is
    #: slope-1 (near-identity) around the centre, so the isometric metric is preserved for
    #: typical colours and only the rare extremes are softly compressed:
    #:   L* = lab_center + lab_lspan*tanh(zL/lab_lspan);  a* = lab_ab*tanh(za/lab_ab); ...
    lab_center: float = 50.0
    lab_lspan: float = 50.0           # L* spans (center-lspan, center+lspan) = (0,100)
    lab_ab: float = 90.0              # comfortable a*/b* perceptual bound


@dataclass
class DataConfig:
    jd_start: float = DEFAULT_JD_START
    jd_end: float = DEFAULT_JD_END
    lat_min: float = -90.0
    lat_max: float = 90.0
    lon_min: float = -180.0
    lon_max: float = 180.0
    batch: int = 2048                 # random skies drawn per step
    seed: int = 0


@dataclass
class TrainConfig:
    lr: float = 1e-4                  # SIRENs like a small LR
    lr_min: float = 1e-6
    weight_decay: float = 0.0
    warmup_steps: int = 500
    max_steps: int = 40_000
    grad_clip: float = 1.0
    #: colour-distance gain: target ||dLab|| = color_scale * ||dSky||. 1.0 = the literal
    #: isometry; a larger value fills the Lab gamut for a more vivid globe.
    color_scale: float = 20.0
    #: gauge anchor pulling the mean colour toward neutral L*=60 (an isometric loss is
    #: translation-free, so this just keeps the output inside the displayable range).
    anchor_weight: float = 0.05
    device: str = ""                  # "" -> auto
    num_workers: int = 0
    out_dir: str = "version6/checkpoints"
    save_every: int = 2000
    log_every: int = 25
    seed: int = 0


def _only_known(dc, d: dict) -> dict:
    known = {f.name for f in fields(dc)}
    return {k: v for k, v in d.items() if k in known}


@dataclass
class V6Config:
    siren: SirenConfig = field(default_factory=SirenConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "V6Config":
        return cls(
            siren=SirenConfig(**_only_known(SirenConfig, d.get("siren", {}))),
            data=DataConfig(**_only_known(DataConfig, d.get("data", {}))),
            train=TrainConfig(**_only_known(TrainConfig, d.get("train", {}))),
        )
