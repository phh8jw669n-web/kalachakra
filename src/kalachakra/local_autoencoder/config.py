"""Configuration dataclasses for the Local Sky Autoencoder (train_v4).

A purely physics-and-kinematics-driven autoencoder: it compresses the localized
Local Sky Matrix of the ten primary bodies into a 3-channel OKLab colour bottleneck
and reconstructs the matrix from that colour. No astrology, no grids, no heuristics
-- only PyTorch, pyswisseph and linear algebra.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .. import constants as C

#: Bodies and raw feature width of the Local Sky Matrix.
N_BODIES: int = 10
RAW_FEATURES: int = 8          # per body: az, alt, ecl_lon, ecl_lat, ang_vel, dist, log_mass, phase
#: Rows of the reconstruction target = 10 bodies + 1 <OBSERVER> row.
N_TOKENS: int = N_BODIES + 1   # 11


@dataclass
class ModelConfig:
    n_bodies: int = N_BODIES
    raw_features: int = RAW_FEATURES
    d_model: int = 128
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 256
    dropout: float = 0.0
    pool: str = "observer"          # "observer" (the <OBSERVER> token) | "gap"
    decoder_hidden: tuple[int, ...] = (64, 256)   # 3 -> 64 -> 256 -> 11*8


@dataclass
class DataConfig:
    start_jd: float = C.KALI_YUGA_EPOCH_JD
    end_jd: float = C.KALI_YUGA_EPOCH_JD + C.TIMELINE_YEARS * C.DAYS_PER_YEAR
    seed: int = 0


@dataclass
class TrainConfig:
    lr: float = 3e-4
    weight_decay: float = 1e-2
    warmup_steps: int = 200
    max_steps: int = 20_000
    batch_size: int = 64
    grad_clip: float = 1.0
    amp: bool = True
    device: str = ""                # "" -> auto (mps/cuda/cpu)
    num_workers: int = 0
    out_dir: str = "checkpoints/local_sky"
    save_every: int = 1000
    log_every: int = 20
    seed: int = 0


@dataclass
class LocalSkyConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "LocalSkyConfig":
        return cls(
            model=ModelConfig(**{**asdict(ModelConfig()), **d.get("model", {})}),
            data=DataConfig(**{**asdict(DataConfig()), **d.get("data", {})}),
            train=TrainConfig(**{**asdict(TrainConfig()), **d.get("train", {})}),
        )
