"""Configuration for the version9 Topocentric Self-Attention engine. Self-contained."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields

from .ephemeris import J2000, N_BODIES

_HALF_SPAN_DAYS = 5000.0 * 365.25
DEFAULT_JD_START = J2000 - _HALF_SPAN_DAYS
DEFAULT_JD_END = J2000 + _HALF_SPAN_DAYS


@dataclass
class AttnConfig:
    """The micro self-attention model. 11 body tokens x 3 dims (N,E,Z) -> L*a*b*."""
    n_bodies: int = N_BODIES          # 11 tokens
    token_dim: int = 3                # [North, East, Zenith] per body
    d_model: int = 32                 # embedding / attention width (single head: d_k = d_model)
    d_ff: int = 64                    # per-token feed-forward hidden
    d_head: int = 32                  # output-head hidden
    n_blocks: int = 2                 # stacked attention+FFN blocks
    #: horizon-visibility attention prior. Every attention score for key body j (and its pool
    #: weight) gets + vis_bias * zenith_j, so above-horizon bodies dominate and below-horizon
    #: ones are suppressed — observer-dependent, horizon-gated attention. Learned Q.K content
    #: modulates on top. This is a fixed structural prior (the isometric objective alone does
    #: not reward peaky attention, so the domain physics is baked in rather than hoped for).
    vis_bias: float = 3.0
    #: pure-chroma head: the model outputs only a*,b* = ab*tanh(z) — no luminance. lab_l is a
    #: FIXED neutral lightness supplied at render time so the globe is a constant-brightness
    #: chromatic energy field (raise it toward ~65 for a more luminous "glow"; higher chroma at
    #: low L* gets gamut-compressed and reads muted).
    lab_l: float = 50.0
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
    lr: float = 3e-4
    lr_min: float = 1e-6
    weight_decay: float = 1e-4
    warmup_steps: int = 500
    max_steps: int = 40_000
    grad_clip: float = 1.0
    gamma: float = 32.0               # colour scale: ||dLab|| = gamma * d_sky
    #: The observer-dependent target distance. d_sky = w_local*d_local + w_rel*d_rel, where
    #: d_rel uses HORIZON-GATED chords R_ij = g_i*g_j*(v_i.v_j), g_b = sigmoid(gate_k*zenith_b).
    #: Gating is what makes the relational term observer-dependent (a conjunction overhead
    #: counts, the same conjunction below the horizon does not) — see losses.py / instructions.
    w_local: float = 0.5
    w_rel: float = 0.5
    gate_k: float = 8.0               # horizon-gate steepness (bigger = sharper coastlines)
    anchor_weight: float = 0.05
    device: str = ""
    num_workers: int = 0
    out_dir: str = "version9/checkpoints"
    save_every: int = 2000
    log_every: int = 25
    seed: int = 0


def _only_known(dc, d: dict) -> dict:
    known = {f.name for f in fields(dc)}
    return {k: v for k, v in d.items() if k in known}


@dataclass
class V9Config:
    attn: AttnConfig = field(default_factory=AttnConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "V9Config":
        return cls(
            attn=AttnConfig(**_only_known(AttnConfig, d.get("attn", {}))),
            data=DataConfig(**_only_known(DataConfig, d.get("data", {}))),
            train=TrainConfig(**_only_known(TrainConfig, d.get("train", {}))),
        )
