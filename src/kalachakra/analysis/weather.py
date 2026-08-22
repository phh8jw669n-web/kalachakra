"""
Cosmic-weather engine — objective geometric energy signatures (blueprint §1.2, §6).

This is the substantive, *real* output of the system: given a real timestamp (and
optionally a location), it computes objective quantities directly from real
planetary geometry — no trained network required, no text, no interpretation.

Signatures produced:

* **Harmonic resonance** — constructive interference from smooth angular phasing
  (bodies near 0/60/120 deg): low-friction energy corridors.
* **Structural tension** — destructive interference from hard-angle collisions
  (bodies near 90/180 deg): high shear.
* **Geometric potential** — circular concentration of the bodies (stelliums /
  mass convergence), R in [0, 1].
* **Temporal shear** — rate of change of the configuration (eclipses, retrograde
  stations, rapid phase transitions).
* **Eclipse proximity** — Sun-Moon alignment coincident with a lunar node.
* **Local intensity** — the global signature localized to each Earth node via the
  analytical horizon projection (which bodies are angular / rising where).

The pure-geometry functions take longitude arrays and are fully unit-tested;
the ``frame_*`` / ``weather_map`` wrappers pull real positions from the ephemeris.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import constants as C
from ..ephemeris import bodies, global_state
from ..grid.geodesic import Grid

# Relative influence of each entity (index matches ephemeris.bodies.ENTITIES).
# Ayanamsha (index 9) is the precession vector, not a body -> weight 0.
BODY_WEIGHTS = np.array([1.0, 1.0, 0.5, 0.5, 0.7, 0.9, 0.9, 0.6, 0.6, 0.0])

# Aspect definitions: name -> (exact angle in degrees, kind).
CONSTRUCTIVE = {"conjunction": 0.0, "sextile": 60.0, "trine": 120.0}
DESTRUCTIVE = {"square": 90.0, "opposition": 180.0}

DEFAULT_ORB_DEG = 6.0  # angular tolerance (Gaussian std) around an exact aspect


def separations_deg(lons_rad: np.ndarray) -> np.ndarray:
    """Pairwise angular separations in degrees, folded to [0, 180]."""
    lons = np.rad2deg(np.asarray(lons_rad, dtype=np.float64)) % 360.0
    diff = np.abs(lons[:, None] - lons[None, :]) % 360.0
    return np.minimum(diff, 360.0 - diff)


def _aspect_kernel(sep: np.ndarray, angle: float, orb: float) -> np.ndarray:
    """Gaussian proximity of a separation to an exact aspect angle, in [0, 1]."""
    return np.exp(-0.5 * ((sep - angle) / orb) ** 2)


def aspect_field(lons_rad: np.ndarray, weights: np.ndarray = BODY_WEIGHTS,
                 orb: float = DEFAULT_ORB_DEG) -> dict:
    """Resonance and tension scalars plus the per-body activation vector.

    Each ordered body pair contributes ``w_i * w_j * kernel`` for every aspect;
    constructive aspects accumulate into resonance, destructive into tension.
    """
    sep = separations_deg(lons_rad)
    n = sep.shape[0]
    w = np.asarray(weights, dtype=np.float64)
    wpair = np.outer(w, w)
    upper = np.triu(np.ones((n, n)), k=1)  # count each unordered pair once

    resonance_mat = np.zeros((n, n))
    for angle in CONSTRUCTIVE.values():
        resonance_mat += _aspect_kernel(sep, angle, orb)
    tension_mat = np.zeros((n, n))
    for angle in DESTRUCTIVE.values():
        tension_mat += _aspect_kernel(sep, angle, orb)

    resonance_mat *= wpair * upper
    tension_mat *= wpair * upper

    # Per-body activation: how strongly each body participates in any aspect.
    activation = (resonance_mat + resonance_mat.T
                  + tension_mat + tension_mat.T).sum(axis=1)

    return {
        "resonance": float(resonance_mat.sum()),
        "tension": float(tension_mat.sum()),
        "resonance_matrix": resonance_mat,
        "tension_matrix": tension_mat,
        "activation": activation,
    }


def stellium_concentration(lons_rad: np.ndarray,
                           weights: np.ndarray = BODY_WEIGHTS) -> tuple[float, float]:
    """Weighted circular concentration R in [0, 1] and mean longitude (rad).

    R -> 1 means the bodies cluster tightly in longitude (a stellium / geometric
    mass convergence); R -> 0 means they are evenly spread.
    """
    lons = np.asarray(lons_rad, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    vec = np.sum(w * np.exp(1j * lons))
    total = w.sum()
    R = float(np.abs(vec) / total) if total else 0.0
    mean_lon = float(np.angle(vec))
    return R, mean_lon


def eclipse_state(lons_rad: np.ndarray, orb_deg: float = 12.0) -> dict:
    """Detect Sun-Moon-node alignments (solar / lunar eclipse proximity)."""
    i = bodies.index_of
    sun, moon, rahu = lons_rad[i("Sun")], lons_rad[i("Moon")], lons_rad[i("Rahu")]

    def sep_deg(a, b):
        d = abs(np.rad2deg(a - b)) % 360.0
        return min(d, 360.0 - d)

    sun_moon = sep_deg(sun, moon)
    moon_node = min(sep_deg(moon, rahu), abs(sep_deg(moon, rahu) - 180.0))
    solar = _aspect_kernel(np.array(sun_moon), 0.0, orb_deg) * \
        _aspect_kernel(np.array(moon_node), 0.0, orb_deg)
    lunar = _aspect_kernel(np.array(sun_moon), 180.0, orb_deg) * \
        _aspect_kernel(np.array(moon_node), 0.0, orb_deg)
    return {
        "solar_proximity": float(solar),
        "lunar_proximity": float(lunar),
        "is_eclipse": bool(max(solar, lunar) > 0.5),
        "sun_moon_sep_deg": float(sun_moon),
        "moon_node_sep_deg": float(moon_node),
    }


def dominant_aspects(lons_rad: np.ndarray, weights: np.ndarray = BODY_WEIGHTS,
                     orb: float = DEFAULT_ORB_DEG, top: int = 5) -> list[dict]:
    """List the strongest exact-ish aspects currently in force (real geometry)."""
    sep = separations_deg(lons_rad)
    n = sep.shape[0]
    names = bodies.NAMES
    all_aspects = {**CONSTRUCTIVE, **DESTRUCTIVE}
    found = []
    for a in range(n):
        for b in range(a + 1, n):
            if weights[a] == 0 or weights[b] == 0:
                continue
            for aname, angle in all_aspects.items():
                k = float(_aspect_kernel(np.array(sep[a, b]), angle, orb))
                if k > 0.4:
                    found.append({
                        "bodies": (names[a], names[b]),
                        "aspect": aname,
                        "separation_deg": float(sep[a, b]),
                        "strength": k * weights[a] * weights[b],
                        "kind": "constructive" if aname in CONSTRUCTIVE else "destructive",
                    })
    found.sort(key=lambda d: d["strength"], reverse=True)
    return found[:top]


# Only true planets retrograde and station; the Sun/Moon never do, and the lunar
# nodes are always slow, so neither belongs in station detection.
_STATIONABLE = ("Mercury", "Venus", "Mars", "Jupiter", "Saturn")


def stations(global_frame: np.ndarray,
             threshold_deg_per_day: float = 0.05) -> list[str]:
    """Planets near a retrograde/direct station (|longitude speed| ~ 0)."""
    speeds = np.rad2deg(np.abs(global_frame[:, 3]))  # lam_dot -> deg/day
    out = []
    for name in _STATIONABLE:
        i = bodies.index_of(name)
        if speeds[i] < threshold_deg_per_day:
            out.append(name)
    return out


@dataclass
class FrameSignature:
    """Objective geometric weather for one instant."""

    jd: float
    resonance: float
    tension: float
    net_interference: float
    potential: float           # stellium concentration R
    mean_longitude_deg: float
    eclipse: dict
    stations: list[str]
    dominant_aspects: list[dict] = field(default_factory=list)
    activation: np.ndarray | None = None


def frame_signature(jd_ut: float, orb: float = DEFAULT_ORB_DEG) -> FrameSignature:
    """Compute the full real weather signature for a timestamp."""
    g = global_state.global_state_frame(jd_ut)
    lons = np.arctan2(g[:, 1], g[:, 0])
    af = aspect_field(lons, orb=orb)
    R, mean_lon = stellium_concentration(lons)
    return FrameSignature(
        jd=jd_ut,
        resonance=af["resonance"],
        tension=af["tension"],
        net_interference=af["resonance"] - af["tension"],
        potential=R,
        mean_longitude_deg=float(np.rad2deg(mean_lon) % 360.0),
        eclipse=eclipse_state(lons),
        stations=stations(g),
        dominant_aspects=dominant_aspects(lons, orb=orb),
        activation=af["activation"],
    )


def temporal_shear(jd_ut: float, dt_days: float = 1.0 / 24.0,
                   orb: float = DEFAULT_ORB_DEG) -> float:
    """Rate of change of the (resonance, tension, R) state, per day.

    Central finite difference across +/- dt. Spikes at eclipses and stations.
    """
    def phi(jd):
        s = frame_signature(jd, orb=orb)
        return np.array([s.resonance, s.tension, s.potential])
    return float(np.linalg.norm(phi(jd_ut + dt_days) - phi(jd_ut - dt_days))
                 / (2.0 * dt_days))


def local_intensity(field: np.ndarray, activation: np.ndarray,
                    weights: np.ndarray = BODY_WEIGHTS) -> np.ndarray:
    """Per-node intensity: globally-active bodies weighted by local angularity.

    ``field`` is E(t, s) of shape ``(N, B, 5)``. A body contributes where it is
    near the local Ascendant (cos(delta phi) ~ 1) and above the horizon, scaled
    by how aspected it is globally (``activation``) and its influence weight.
    """
    cos_dphi = field[..., 3]                 # (N, B): near 1 at the Ascendant
    above = 0.5 * (field[..., 2] + 1.0)      # (N, B): 0 below horizon, 1 at zenith
    angularity = 0.5 * (cos_dphi + 1.0) * (0.5 + 0.5 * above)  # (N, B) in [0,1]
    strength = angularity * (weights * (1.0 + activation))[None, :]
    return strength.sum(axis=1)              # (N,)


def weather_map(jd_ut: float, grid: Grid, dt_days: float = 1.0 / 24.0,
                orb: float = DEFAULT_ORB_DEG) -> dict:
    """Real per-node potential and shear fields for the whole mesh.

    Potential = local intensity of the current geometry; shear = its temporal
    derivative (finite difference dt_days apart). Both are real functions of real
    planetary positions.
    """
    from ..projection import spatial

    g0 = global_state.global_state_frame(jd_ut)
    g1 = global_state.global_state_frame(jd_ut + dt_days)
    lons0 = np.arctan2(g0[:, 1], g0[:, 0])
    act0 = aspect_field(lons0, orb=orb)["activation"]
    lons1 = np.arctan2(g1[:, 1], g1[:, 0])
    act1 = aspect_field(lons1, orb=orb)["activation"]

    field0 = spatial.project(g0, jd_ut, grid)
    field1 = spatial.project(g1, jd_ut + dt_days, grid)
    potential = local_intensity(field0, act0)
    potential1 = local_intensity(field1, act1)
    shear = np.abs(potential1 - potential) / dt_days

    sig = frame_signature(jd_ut, orb=orb)
    return {
        "jd": jd_ut,
        "potential": potential,             # (N,)
        "shear": shear,                     # (N,)
        "signature": sig,
    }
