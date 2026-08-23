"""
Dynamic velocity scaling and the textless geometric news radar (blueprint §6).

As the user scrubs the timeline, the engine self-regulates:

* **Temporal stride / velocity scaling** — decimate frames so a query streams a
  bounded number of visible points regardless of span (hours .. ten millennia).
* **Four-band frequency mixer** — Micro (24 s diurnal), Fast (Moon/Mercury/Venus
  synodic), Cyclic (Mars/Sun/nodes), Macro (outer planets / precession). Band
  gains attenuate high frequencies during fast macro scrubbing and restore them
  on deceleration, like a spectral equalizer.
* **Dynamic significance threshold** — the rarity percentile that flags an event
  tightens toward 99.99 at deep-time zoom and relaxes for short windows.
* **News cards** — pure-geometry event payloads (ids, Cartesian body vectors,
  applying/separating, culmination angles, rarity percentile); no text folklore.

Pure numpy; fully tested.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import weather
from ..ephemeris import bodies

# Characteristic period of each band, in frames (1 frame = 24 s).
_FRAMES_PER_DAY = 3600
BAND_PERIOD_FRAMES = {
    "micro": 1.0,                          # diurnal horizon crossing (24 s)
    "fast": 29.5 * _FRAMES_PER_DAY,        # lunar synodic month
    "cyclic": 2.135 * 365.25 * _FRAMES_PER_DAY,   # Mars synodic (~780 d)
    "macro": 29.4 * 365.25 * _FRAMES_PER_DAY,     # Saturn orbital period
}
BAND_ORDER = ("micro", "fast", "cyclic", "macro")


def temporal_stride(span_frames: int, target_points: int = 1000) -> int:
    """Frame decimation stride so a span streams ~``target_points`` samples."""
    span_frames = max(1, int(span_frames))
    if span_frames <= target_points:
        return 1
    return int(np.ceil(span_frames / target_points))


def band_gains(stride: int, sharpness: float = 1.5,
               offset: float = 1.0) -> dict[str, float]:
    """Per-band gain in [0, 1] for a given temporal stride.

    A band whose period is well below the sampling stride would alias, so it is
    attenuated; bands whose period is at or above the stride pass at (near) full
    gain. Uses a logistic in log-period space; ``offset`` shifts the transition
    so a band at ``period == stride`` still mostly passes (only clearly
    sub-stride periods are suppressed).
    """
    stride = max(1.0, float(stride))
    log_s = np.log10(stride)
    gains = {}
    for name, period in BAND_PERIOD_FRAMES.items():
        x = np.log10(period) - log_s + offset
        gains[name] = float(1.0 / (1.0 + np.exp(-sharpness * x)))
    return gains


def significance_percentile(span_years: float,
                            lo_span: float = 1.0, hi_span: float = 10256.0,
                            lo_p: float = 95.0, hi_p: float = 99.99) -> float:
    """Rarity percentile that flags an event, tightening with span (log scale)."""
    span_years = max(lo_span, float(span_years))
    t = (np.log(span_years) - np.log(lo_span)) / (np.log(hi_span) - np.log(lo_span))
    t = float(np.clip(t, 0.0, 1.0))
    return lo_p + t * (hi_p - lo_p)


def is_applying(lons: np.ndarray, speeds: np.ndarray, a: int, b: int,
                orb: float = weather.DEFAULT_ORB_DEG) -> bool:
    """Whether bodies ``a`` and ``b`` are applying (separation shrinking toward
    the nearest exact aspect) rather than separating."""
    sep = float(weather.separations_deg(lons)[a, b])
    targets = list(weather.CONSTRUCTIVE.values()) + list(weather.DESTRUCTIVE.values())
    nearest = min(targets, key=lambda t: abs(sep - t))
    rel_speed = float(np.rad2deg(speeds[a] - speeds[b]))  # deg/day
    # d|sep-target|/dt < 0  =>  applying.
    return (sep - nearest) * rel_speed < 0.0


@dataclass
class NewsCard:
    """Pure-geometry event payload anchored to a coordinate (blueprint §6)."""

    jd: float
    lat: float
    lng: float
    macro_id: int
    micro_id: int
    rarity: float
    rarity_percentile: float
    applying: bool
    bodies: list[dict] = field(default_factory=list)
    dominant_aspect: dict | None = None

    def to_dict(self) -> dict:
        return {
            "jd": self.jd, "lat": self.lat, "lng": self.lng,
            "macro": self.macro_id, "micro": self.micro_id,
            "rarity": self.rarity, "rarity_percentile": self.rarity_percentile,
            "applying": self.applying, "bodies": self.bodies,
            "dominant_aspect": self.dominant_aspect,
        }


def build_news_card(jd: float, lat: float, lng: float, global_frame: np.ndarray,
                    macro_id: int, micro_id: int, rarity: float,
                    rarity_percentile: float) -> NewsCard:
    """Assemble a textless news card from a G(t) frame and token/rarity metadata.

    ``global_frame`` is the (10, 7) state vector; body Cartesian unit vectors and
    angular velocities come straight from it, so the card is pure geometry.
    """
    lons = np.arctan2(global_frame[:, 1], global_frame[:, 0])
    speeds = global_frame[:, 3]  # lambda_dot (rad/day)
    dom = weather.dominant_aspects(lons)
    applying = False
    dom_payload = None
    if dom:
        top = dom[0]
        ai = bodies.index_of(top["bodies"][0])
        bi = bodies.index_of(top["bodies"][1])
        applying = is_applying(lons, speeds, ai, bi)
        dom_payload = {**top, "applying": applying}

    body_list = []
    for i, name in enumerate(bodies.NAMES):
        if weather.BODY_WEIGHTS[i] == 0:
            continue
        body_list.append({
            "name": name,
            "unit_vector": global_frame[i, :3].astype(float).tolist(),
            "angular_velocity_deg_per_day": float(np.rad2deg(global_frame[i, 3])),
        })

    return NewsCard(
        jd=float(jd), lat=float(lat), lng=float(lng),
        macro_id=int(macro_id), micro_id=int(micro_id),
        rarity=float(rarity), rarity_percentile=float(rarity_percentile),
        applying=applying, bodies=body_list, dominant_aspect=dom_payload,
    )
