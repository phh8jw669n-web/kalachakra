"""Configuration for the version5 Sky-Energy Autoencoder.

Small, plain dataclasses (serialised into every checkpoint) so a trained model and
its ONNX export always know their own geometry.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from kalachakra.ephemeris import calendar as _cal

# -- fixed geometry ----------------------------------------------------------
# 12 bodies: Sun..Pluto (0..9) + Mean Node (10) + True Node (11).
N_BODIES: int = 12
# per body: altitude, azimuth, ecliptic longitude, ecliptic latitude, house offset,
# velocity (the "True Astrological Shape").
RAW_FEATURES: int = 6
# the <OBSERVER> token ingests the high-frequency geographic anchors Asc, MC, Vertex.
OBS_FEATURES: int = 3
RECON_FEATURES: int = 4            # reconstruction target per body: (sin,cos) of altitude & azimuth
N_TOKENS: int = N_BODIES + 1       # + the data-driven <OBSERVER> token = 13

#: The 24-second astrological quantum (a Vighatika) expressed in days — the finest
#: temporal grid the Monte-Carlo sampler ever lands on.
VIGHATIKA_SECONDS: float = 24.0
VIGHATIKA_DAYS: float = VIGHATIKA_SECONDS / 86_400.0

#: Full 10,256-year span, as Julian Days (roughly 3101 BCE .. 7155 CE).
DEFAULT_START_JD: float = _cal.parse_datetime("-3101-02-18T00:00:00")
DEFAULT_END_JD: float = _cal.parse_datetime("7155-02-18T00:00:00")


@dataclass
class ModelConfig:
    n_bodies: int = N_BODIES
    raw_features: int = RAW_FEATURES
    obs_features: int = OBS_FEATURES
    recon_features: int = RECON_FEATURES
    d_model: int = 112
    nhead: int = 8
    num_layers: int = 3
    dim_feedforward: int = 288
    dropout: float = 0.0
    pool: str = "observer"                       # "observer" (the token) | "gap"
    decoder_hidden: tuple[int, ...] = (64, 256)  # 3 -> 64 -> 256 -> 12*4


@dataclass
class DataConfig:
    start_jd: float = DEFAULT_START_JD
    end_jd: float = DEFAULT_END_JD
    #: Locations sampled per timestamp = the broadcast batch of a single ephemeris
    #: query (see the "single query rule", PRD page 4).
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
    mass_weighting: bool = True
    out_dir: str = "version5/checkpoints"
    save_every: int = 2000
    log_every: int = 25
    seed: int = 0


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
            model=ModelConfig(**{**asdict(ModelConfig()), **d.get("model", {})}),
            data=DataConfig(**{**asdict(DataConfig()), **d.get("data", {})}),
            train=TrainConfig(**{**asdict(TrainConfig()), **d.get("train", {})}),
        )
