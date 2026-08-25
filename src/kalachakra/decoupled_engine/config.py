"""Configuration dataclasses for the decoupled projection engine.

The engine separates *celestial geometry* (the Sky Encoder, which compresses the
ten-body ephemeris state into one global tension vector) from *terrestrial spatial
mapping* (the Earth Lens Decoder, a coordinate-based implicit field that turns the
tension vector plus a continuous lat/lon into a perceptual colour). Nothing here is
discretised onto a fixed mesh; every knob is a continuous hyper-parameter so a run
is fully reproducible from a single frozen config.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .. import constants as C

#: The ten primary bodies this engine models, in Swiss-Ephemeris id order 0..9.
N_BODIES: int = 10
#: Per-body input features: [sin lon, cos lon, sin lat, cos lat, lon_velocity].
BODY_FEATURES: int = 5


@dataclass
class SkyEncoderConfig:
    n_bodies: int = N_BODIES
    in_features: int = BODY_FEATURES
    d_model: int = 128
    nhead: int = 8                     # >= 8 heads for all-to-all aspect attention
    num_layers: int = 4
    dim_feedforward: int = 256
    dropout: float = 0.0
    tension_dim: int = 512             # the global tension vector width
    normalize_output: bool = True      # L2-normalise the tension vector
    grad_checkpoint: bool = False


@dataclass
class EarthLensConfig:
    tension_dim: int = 512
    #: coordinates are lifted (lat, lon) -> unit 3-vector before Fourier encoding,
    #: which removes the +/-180 deg seam (continuous on the sphere).
    coord_dim: int = 3
    num_fourier: int = 64              # random Fourier feature count (-> 2*num channels)
    fourier_scale: float = 8.0         # std of the RFF frequency matrix
    learnable_fourier: bool = False
    hidden: int = 256
    n_blocks: int = 4                  # residual MLP blocks
    activation: str = "gauss"          # "gauss" | "sine"
    gauss_sigma: float = 0.1           # width of the Gaussian activation
    sine_omega0: float = 30.0          # SIREN first-layer frequency
    out_channels: int = 3              # OKLab: (L, a, b)
    bound_output: bool = True          # squash to the OKLab domain (L in [0,1])
    ab_scale: float = 0.4              # half-range of the a/b chromaticity channels


@dataclass
class DataConfig:
    """Continuous temporal-slice streaming + terrestrial coordinate sampling."""
    start_jd: float = C.KALI_YUGA_EPOCH_JD
    end_jd: float = C.KALI_YUGA_EPOCH_JD + C.TIMELINE_YEARS * C.DAYS_PER_YEAR
    temporal_len: int = 3              # consecutive frames per slice (temporal loss)
    stride_seconds: float = 3600.0     # spacing between the frames of a slice
    points_per_frame: int = 1024       # random terrestrial coords per slice
    samples_per_epoch: int = 4096      # slices yielded before an epoch ends
    seed: int = 0

    @property
    def stride_days(self) -> float:
        return self.stride_seconds / C.SECONDS_PER_DAY


@dataclass
class TrainConfig:
    lr: float = 3e-4
    weight_decay: float = 1e-2
    warmup_steps: int = 200
    max_steps: int = 5000
    batch_size: int = 8
    grad_clip: float = 1.0
    # composite-loss weights
    w_geometric: float = 1.0
    w_terrestrial: float = 0.5
    w_temporal: float = 0.25
    geo_temperature: float = 0.1       # contrastive softmax temperature
    geodesic_eps_deg: float = 0.5      # neighbour offset for the smoothness gradient
    amp: bool = True
    device: str = ""                   # "" -> auto (mps/cuda/cpu)
    out_dir: str = "checkpoints/decoupled"
    save_every: int = 250
    log_every: int = 10
    seed: int = 0


@dataclass
class EngineConfig:
    """The full engine: both models + data + training, one serialisable object."""
    sky: SkyEncoderConfig = field(default_factory=SkyEncoderConfig)
    earth: EarthLensConfig = field(default_factory=EarthLensConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self):
        # the decoder must consume exactly what the encoder emits
        if self.earth.tension_dim != self.sky.tension_dim:
            raise ValueError(
                f"tension_dim mismatch: sky={self.sky.tension_dim} "
                f"earth={self.earth.tension_dim}")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EngineConfig":
        return cls(
            sky=SkyEncoderConfig(**d.get("sky", {})),
            earth=EarthLensConfig(**d.get("earth", {})),
            data=DataConfig(**d.get("data", {})),
            train=TrainConfig(**d.get("train", {})),
        )
