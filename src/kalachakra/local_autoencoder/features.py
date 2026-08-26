"""The physics engine: the Local Sky Matrix for the ten primary bodies.

For a sample ``(jd, lat, lon)`` this computes, per body, the eight physical
features the autoencoder compresses -- azimuth, altitude, ecliptic longitude,
ecliptic latitude, angular velocity, distance, log-mass and phase (elongation) --
straight from ``pyswisseph``. No astrology: only ephemeris kinematics + geometry.

Angular features are stored in **radians**; the encoder later expands the cyclic
ones to ``(sin, cos)`` so there is no 0/360-degree discontinuity. Non-angular
features are normalised so gradients do not explode.
"""

from __future__ import annotations

import numpy as np

from ..ephemeris import global_state as gs

#: Swiss-Ephemeris ids 0..9 == Sun, Moon, Mercury, Venus, Mars, Jupiter, Saturn,
#: Uranus, Neptune, Pluto.
BODY_SWE_IDS: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
BODY_NAMES: tuple[str, ...] = (
    "Sun", "Moon", "Mercury", "Venus", "Mars",
    "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto",
)
N_BODIES = len(BODY_SWE_IDS)
N_TOKENS = N_BODIES + 1                      # + the <OBSERVER> row

# -- raw feature layout (columns of the (10, 8) matrix) ----------------------
COL_AZ, COL_ALT, COL_ECL_LON, COL_ECL_LAT = 0, 1, 2, 3
COL_ANG_VEL, COL_DIST, COL_LOG_MASS, COL_PHASE = 4, 5, 6, 7
#: Columns holding an angle (radians) -> expanded to (sin, cos) for the encoder.
ANGULAR_COLS: tuple[int, ...] = (COL_AZ, COL_ALT, COL_ECL_LON, COL_ECL_LAT, COL_PHASE)
SCALAR_COLS: tuple[int, ...] = (COL_ANG_VEL, COL_DIST, COL_LOG_MASS)
#: Columns that wrap at 0/360 deg -> reconstructed with a circular (wrap-safe) loss.
WRAP_COLS: tuple[int, ...] = (COL_AZ, COL_ECL_LON)
#: Encoder input width after (sin, cos) expansion of the angular columns.
ENCODER_IN = 2 * len(ANGULAR_COLS) + len(SCALAR_COLS)     # 13

# -- normalisation constants -------------------------------------------------
#: log10(mass / Earth mass) for the ten bodies (bounded so gradients stay sane
#: after dividing by LOG_MASS_SCALE): Sun ~5.5, Jupiter ~2.5, Pluto ~-2.7.
LOG_MASS_RAW = np.array(
    [5.52, -1.91, -1.26, -0.09, -0.97, 2.50, 1.98, 1.16, 1.23, -2.66],
    dtype=np.float64)
LOG_MASS_SCALE: float = 6.0
ANG_VEL_SCALE: float = 15.0                 # deg/day; ~Moon's speed -> ~O(1)

#: Per-feature reconstruction emphasis. Azimuth & altitude are weighted hardest --
#: getting them right proves the net mapped the global geometry to the local frame.
FEATURE_W = np.array([3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0], dtype=np.float64)

#: Mass weighting: bounded, increasing in mass (Sun/Jupiter high, Moon/Pluto low).
_lm_min, _lm_max = float(LOG_MASS_RAW.min()), float(LOG_MASS_RAW.max())
MASS_W = 0.5 + 2.0 * (LOG_MASS_RAW - _lm_min) / (_lm_max - _lm_min)   # ~[0.5, 2.5]

#: Proximity (inverse-distance) weighting bounds.
PROX_SOFTEN: float = 0.25
PROX_CLIP: tuple[float, float] = (0.2, 4.0)
#: Weight of the <OBSERVER> row (the local geographic frame).
OBSERVER_W: float = 1.0


# ---------------------------------------------------------------------------
# physics
# ---------------------------------------------------------------------------
#: Columns of the compact "ecliptic state" cache row, per body.
ECL_COLS = ("lon_deg", "lat_deg", "lon_speed_deg_per_day", "dist_au")


def ecliptic_state(jd_ut: float) -> np.ndarray:
    """The expensive part: ``(10, 4)`` geocentric ecliptic state via ``calc_ut``.

    Columns ``[lon_deg, lat_deg, lon_speed_deg_per_day, dist_au]``. This is the only
    step that queries the ephemeris files; caching it (see :mod:`.sky_cache`) removes
    the training loop's dependence on ``calc_ut`` entirely.
    """
    gs._require_swe()
    flags = gs._calc_flags()                        # backend + speed flag
    jd = float(jd_ut)
    ecl = np.empty((N_BODIES, 4), dtype=np.float64)
    for i, sid in enumerate(BODY_SWE_IDS):
        v = gs.swe.calc_ut(jd, sid, flags)[0]
        ecl[i] = (v[0], v[1], v[3], v[2])           # lon, lat, lon_speed, dist
    return ecl


def features_from_ecliptic(jd_ut: float, lat_deg: float, lon_deg: float,
                           ecl: np.ndarray):
    """Build the ``(10, 8)`` Local Sky Matrix from a precomputed ecliptic state.

    ``ecl`` is ``(10, 4)`` ``[lon_deg, lat_deg, lon_speed_deg_per_day, dist_au]``.
    Only ``swe_azalt`` (a pure coordinate transform, no ephemeris query) is used
    here, so this is cheap and backend-independent. Returns
    ``(features float32 (10,8), distance_au float64 (10,))``.
    """
    gs._require_swe()
    jd = float(jd_ut)
    geopos = (float(lon_deg), float(lat_deg), 0.0)
    feat = np.zeros((N_BODIES, 8), dtype=np.float64)
    dist_au = np.zeros(N_BODIES, dtype=np.float64)
    sun_lon, sun_lat = np.deg2rad(ecl[0, 0]), np.deg2rad(ecl[0, 1])

    for i in range(N_BODIES):
        lon_deg_b, lat_deg_b, lon_sp, dist = ecl[i]
        az, true_alt, _app = gs.swe.azalt(
            jd, gs.swe.ECL2HOR, geopos, 0.0, 0.0,
            (float(lon_deg_b), float(lat_deg_b), float(dist)))
        lam, bet = np.deg2rad(lon_deg_b), np.deg2rad(lat_deg_b)
        # elongation: Sun-Earth-Planet angle (geocentric angular separation to Sun)
        cos_e = (np.sin(bet) * np.sin(sun_lat)
                 + np.cos(bet) * np.cos(sun_lat) * np.cos(lam - sun_lon))
        phase = float(np.arccos(np.clip(cos_e, -1.0, 1.0)))
        feat[i, COL_AZ] = np.deg2rad(az)
        feat[i, COL_ALT] = np.deg2rad(true_alt)
        feat[i, COL_ECL_LON] = lam
        feat[i, COL_ECL_LAT] = bet
        feat[i, COL_ANG_VEL] = lon_sp / ANG_VEL_SCALE
        feat[i, COL_DIST] = np.log10(max(dist, 1e-6))
        feat[i, COL_LOG_MASS] = LOG_MASS_RAW[i] / LOG_MASS_SCALE
        feat[i, COL_PHASE] = phase
        dist_au[i] = dist
    return feat.astype(np.float32), dist_au


def local_sky_matrix(jd_ut: float, lat_deg: float, lon_deg: float):
    """Compute the ``(10, 8)`` Local Sky Matrix and per-body distance (AU), live."""
    return features_from_ecliptic(jd_ut, lat_deg, lon_deg, ecliptic_state(jd_ut))


def observer_row(jd_ut: float, lat_deg: float, lon_deg: float) -> np.ndarray:
    """The 11th (<OBSERVER>) reconstruction row: the local geographic frame.

    ``[sin lat, cos lat, sin lon, cos lon, sin LST, cos LST, 0, 0]`` -- all in
    ``[-1, 1]``. Forces the OKLab bottleneck to encode *where* the observer stands
    and the local sidereal time, not just the global sky.
    """
    lat, lon = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    lst_deg = (gs.swe.sidtime(float(jd_ut)) * 15.0 + lon_deg) % 360.0
    lst = np.deg2rad(lst_deg)
    return np.array(
        [np.sin(lat), np.cos(lat), np.sin(lon), np.cos(lon),
         np.sin(lst), np.cos(lst), 0.0, 0.0], dtype=np.float32)


def build_weight(dist_au: np.ndarray) -> np.ndarray:
    """The physics weight tensor ``(11, 8)`` = mass x proximity x feature."""
    prox = np.clip(1.0 / (dist_au + PROX_SOFTEN), *PROX_CLIP)          # (10,)
    body_w = (MASS_W * prox)[:, None] * FEATURE_W[None, :]             # (10,8)
    obs_w = OBSERVER_W * np.ones((1, 8), dtype=np.float64)
    return np.concatenate([body_w, obs_w], axis=0).astype(np.float32)  # (11,8)


def sample_tensors_from_ecl(jd_ut: float, lat_deg: float, lon_deg: float,
                            ecl: np.ndarray):
    """Assemble ``(features (10,8), target (11,8), weight (11,8))`` from an ecliptic
    state — the shared core of the live and sky-cache data paths."""
    feat, dist_au = features_from_ecliptic(jd_ut, lat_deg, lon_deg, ecl)
    obs = observer_row(jd_ut, lat_deg, lon_deg)
    target = np.concatenate([feat, obs[None, :]], axis=0)             # (11,8)
    weight = build_weight(dist_au)
    return feat, target, weight


def sample_tensors(jd_ut: float, lat_deg: float, lon_deg: float):
    """One live sample -> ``(features (10,8), target (11,8), weight (11,8))``."""
    return sample_tensors_from_ecl(jd_ut, lat_deg, lon_deg, ecliptic_state(jd_ut))


def circular_mask() -> np.ndarray:
    """``(11, 8)`` bool: True where the reconstruction target is a wrapping angle
    (body rows, azimuth & ecliptic longitude) -> use the circular loss there."""
    m = np.zeros((N_TOKENS, 8), dtype=bool)
    for c in WRAP_COLS:
        m[:N_BODIES, c] = True
    return m


def sample_sphere(rng: np.random.Generator) -> tuple[float, float]:
    """A random ``(lat_deg, lon_deg)`` uniform over the sphere's area."""
    lat = np.rad2deg(np.arcsin(rng.uniform(-1.0, 1.0)))
    lon = rng.uniform(-180.0, 180.0)
    return float(lat), float(lon)
