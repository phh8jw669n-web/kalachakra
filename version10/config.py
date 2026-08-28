"""Configuration for the version10 Topocentric Self-Attention engine. Self-contained."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields

from .ephemeris import J2000, N_BODIES

_HALF_SPAN_DAYS = 5000.0 * 365.25
DEFAULT_JD_START = J2000 - _HALF_SPAN_DAYS
DEFAULT_JD_END = J2000 + _HALF_SPAN_DAYS


@dataclass
class AttnConfig:
    """The micro self-attention model. 13 tokens x 3 dims (N,E,Z) -> OKLCH chroma."""
    n_bodies: int = N_BODIES          # 13 tokens (11 bodies + ASC + MC)
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
    #: OKLCH pure-chroma head (no luminance): the model outputs polar (C,H) ->
    #: OKLab (a,b) = (C cosH, C sinH), C = okl_cmax*sigmoid(z0), H = z1 (raw radians). okl_l is
    #: the FIXED neutral OKLab lightness supplied only at render time, so the globe is a
    #: constant-lightness, perceptually-uniform chromatic energy field. okl_cmax caps chroma to a
    #: renderable range (0.4 is vivid; lower it if the most saturated hues clip).
    okl_l: float = 0.5
    okl_cmax: float = 0.4


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
    gamma: float = 0.35               # chroma scale: ||d(OKLab a,b)|| = gamma * d_sky. OKLab
                                      # units (~60x < CIELab). ~0.35 keeps most colours inside
                                      # the sRGB gamut (vivid but faithful); raise toward 0.5 for
                                      # a more saturated field at the cost of more gamut clipping.
    #: The observer-dependent target distance. d_sky = w_local*d_local + w_rel*d_rel, where
    #: d_rel uses HORIZON-GATED chords R_ij = g_i*g_j*(v_i.v_j), g_b = sigmoid(gate_k*zenith_b).
    #: Gating is what makes the relational term observer-dependent (a conjunction overhead
    #: counts, the same conjunction below the horizon does not) — see losses.py / instructions.
    w_local: float = 0.5
    w_rel: float = 0.5
    #: Softer horizon gate in v10 (was 8.0). A gentle physical falloff for angular proximity
    #: avoids a near-discontinuous target — the model gets its SHARP structure from the fast
    #: ASC/MC tokens (which cross zeniths/horizons rapidly with geography), not from a steep gate
    #: that would push the net into Gibbs-style spatial ringing.
    gate_k: float = 3.0
    #: Total-variation smoothness prior. Penalises squared colour change between geographically
    #: neighbouring observers (same jd), so the model learns genuine localized tension from the
    #: tokens rather than hallucinating high-frequency noise. 0 disables it.
    tv_weight: float = 0.05
    tv_delta_deg: float = 0.75        # neighbour offset (deg) used to probe the spatial gradient
    anchor_weight: float = 0.05
    device: str = ""
    num_workers: int = 0
    out_dir: str = "version10/checkpoints"
    save_every: int = 2000
    log_every: int = 25
    seed: int = 0


def _only_known(dc, d: dict) -> dict:
    known = {f.name for f in fields(dc)}
    return {k: v for k, v in d.items() if k in known}


@dataclass
class V10Config:
    attn: AttnConfig = field(default_factory=AttnConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "V10Config":
        return cls(
            attn=AttnConfig(**_only_known(AttnConfig, d.get("attn", {}))),
            data=DataConfig(**_only_known(DataConfig, d.get("data", {}))),
            train=TrainConfig(**_only_known(TrainConfig, d.get("train", {}))),
        )
