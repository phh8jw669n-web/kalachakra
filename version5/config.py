"""Configuration for the version5.1 metric-learning encoder.

Small, plain dataclasses (serialised into every checkpoint) so a trained model and
its ONNX export always know their own geometry.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields

from kalachakra.ephemeris import calendar as _cal

# -- fixed geometry (Zero-Redundancy 50-D state) -----------------------------
# 11 ML bodies: Sun..Pluto (0..9) + True Node — the Mean Node is dropped as redundant.
N_ML_BODIES: int = 11
# per body: 3D ecliptic Cartesian unit vector (X,Y,Z) + tanh-normalised velocity (V).
BODY_FEATURES: int = 4
# observer anchors: Ascendant + Midheaven, each as a 3D ecliptic Cartesian unit vector.
OBS_FEATURES: int = 6
# the flat non-redundant physical state fed to the encoder AND to the isometric loss.
STATE_DIM: int = N_ML_BODIES * BODY_FEATURES + OBS_FEATURES     # 44 + 6 = 50
# the model is a sequence of 11 body tokens + 1 observer token.
N_TOKENS: int = N_ML_BODIES + 1                                # 12

#: The 24-second astrological quantum (a Vighatika) expressed in days — the finest
#: temporal grid the Monte-Carlo sampler ever lands on.
VIGHATIKA_SECONDS: float = 24.0
VIGHATIKA_DAYS: float = VIGHATIKA_SECONDS / 86_400.0

#: Full 10,256-year span, as Julian Days (roughly 3101 BCE .. 7155 CE).
DEFAULT_START_JD: float = _cal.parse_datetime("-3101-02-18T00:00:00")
DEFAULT_END_JD: float = _cal.parse_datetime("7155-02-18T00:00:00")


@dataclass
class ModelConfig:
    n_bodies: int = N_ML_BODIES
    body_features: int = BODY_FEATURES
    obs_features: int = OBS_FEATURES
    d_model: int = 112
    nhead: int = 8
    num_layers: int = 3
    dim_feedforward: int = 288
    dropout: float = 0.0
    pool: str = "observer"                       # "observer" (the token) | "gap"

    @property
    def state_dim(self) -> int:
        return self.n_bodies * self.body_features + self.obs_features


@dataclass
class DataConfig:
    start_jd: float = DEFAULT_START_JD
    end_jd: float = DEFAULT_END_JD
    #: Locations sampled per timestamp = the broadcast batch of a single ephemeris query.
    locations_per_step: int = 2048
    seed: int = 0


@dataclass
class TrainConfig:
    lr: float = 3e-4
    lr_min: float = 1e-6              # cosine-decay floor reached at the final step
    weight_decay: float = 1e-2
    warmup_steps: int = 1000
    max_steps: int = 40_000
    grad_clip: float = 1.0
    amp: bool = False
    device: str = ""                             # "" -> auto (mps/cuda/cpu)
    num_workers: int = 0
    out_dir: str = "version5/checkpoints"
    save_every: int = 2000
    log_every: int = 25
    seed: int = 0


def _only_known(dc, d: dict) -> dict:
    """Keep only keys that are real fields of dataclass ``dc`` (tolerate old checkpoints)."""
    known = {f.name for f in fields(dc)}
    return {k: v for k, v in d.items() if k in known}


@dataclass
class V5Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "V5Config":
        return cls(
            model=ModelConfig(**_only_known(ModelConfig, d.get("model", {}))),
            data=DataConfig(**_only_known(DataConfig, d.get("data", {}))),
            train=TrainConfig(**_only_known(TrainConfig, d.get("train", {}))),
        )
