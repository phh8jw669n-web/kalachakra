"""Self-contained analytic topocentric ephemeris — the single source of truth.

For any observer ``(lat, lon)`` and continuous Julian Date ``jd`` this produces the
33-D local-sky tensor: 11 bodies (Sun, Moon, Mercury, Venus, Mars, Jupiter, Saturn,
Uranus, Neptune, Pluto, True Node), each as a **North/East/Up** unit vector in the
observer's horizontal frame.

Pipeline per body:  heliocentric Kepler elements -> geocentric ecliptic -> equatorial
(obliquity) -> local horizontal (hour angle + latitude) -> Cartesian ``(N, E, Up)``.

Everything is closed-form (JPL low-precision Keplerian elements, a short lunar series,
a linear node) so the *identical* maths is transcribed to ``web/ephemeris6.js`` and the
GLSL shader. Accuracy is arc-minute class near the present and degrades smoothly over
millennia — deliberate, since the SIREN only needs a continuous geometric field, not an
almanac. Vectorised over the batch (numpy); the JS/GLSL ports run it per sample/pixel.
"""

from __future__ import annotations

import numpy as np

J2000 = 2451545.0
DEG = np.pi / 180.0

BODY_NAMES = (
    "Sun", "Moon", "Mercury", "Venus", "Mars", "Jupiter",
    "Saturn", "Uranus", "Neptune", "Pluto", "Node",
    "ASC", "MC",                 # v10: the two astrological structural anchors (tokens 11, 12)
)
N_PLANETS_TOK = 11               # Sun..Node have geocentric (observer-independent) directions
N_BODIES = 13                    # + Ascendant + Midheaven (observer-dependent)
STATE_DIM = N_BODIES * 3         # 39
_LAT_CLAMP = 89.99               # keep the topocentric frame / tan(lat) off the exact pole

# JPL "Keplerian Elements for Approximate Positions" (Standish), J2000 + rates/century.
# columns: a[au], e, I[deg], L[deg], long.peri[deg], long.node[deg]
# (Earth row is the Earth-Moon barycentre.)
_PLANET_NAMES = ("Mercury", "Venus", "Earth", "Mars", "Jupiter",
                 "Saturn", "Uranus", "Neptune", "Pluto")
_ELEM = np.array([
    [0.38709927, 0.20563593, 7.00497902, 252.25032350, 77.45779628, 48.33076593],
    [0.72333566, 0.00677672, 3.39467605, 181.97909950, 131.60246718, 76.67984255],
    [1.00000261, 0.01671123, -0.00001531, 100.46457166, 102.93768193, 0.0],
    [1.52371034, 0.09339410, 1.84969142, -4.55343205, -23.94362959, 49.55953891],
    [5.20288700, 0.04838624, 1.30439695, 34.39644051, 14.72847983, 100.47390909],
    [9.53667594, 0.05386179, 2.48599187, 49.95424423, 92.59887831, 113.66242448],
    [19.18916464, 0.04725744, 0.77263783, 313.23810451, 170.95427630, 74.01692503],
    [30.06992276, 0.00859048, 1.77004347, -55.12002969, 44.96476227, 131.78422574],
    [39.48211675, 0.24882730, 17.14001206, 238.92903833, 224.06891629, 110.30393684],
], dtype=np.float64)
_RATE = np.array([
    [0.00000037, 0.00001906, -0.00594749, 149472.67411175, 0.16047689, -0.12534081],
    [0.00000390, -0.00004107, -0.00078890, 58517.81538729, 0.00268329, -0.27769418],
    [0.00000562, -0.00004392, -0.01294668, 35999.37244981, 0.32327364, 0.0],
    [0.00001847, 0.00007882, -0.00813131, 19140.30268499, 0.44441088, -0.29257343],
    [-0.00011607, -0.00013253, -0.00183714, 3034.74612775, 0.21252668, 0.20469106],
    [-0.00125060, -0.00050991, 0.00193609, 1222.49362201, -0.41897216, -0.28867794],
    [-0.00196176, -0.00004397, -0.00242939, 428.48202785, 0.40805281, 0.04240589],
    [0.00026291, 0.00005105, 0.00035372, 218.45945325, -0.32241464, -0.00508664],
    [-0.00031596, 0.00005170, 0.00004818, 145.20780515, -0.04062942, -0.01183482],
], dtype=np.float64)
_EARTH = _PLANET_NAMES.index("Earth")


def _wrap180(deg):
    return (deg + 180.0) % 360.0 - 180.0


def _kepler_E(M, e, iters: int = 6):
    """Solve M = E - e sin E (M, E radians) by Newton — fixed iteration count."""
    E = M + e * np.sin(M)
    for _ in range(iters):
        E = E - (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
    return E


def _heliocentric(T, i: int):
    """Heliocentric J2000-ecliptic Cartesian ``(x,y,z)`` [AU] of planet ``i`` at ``T``.

    ``T`` is centuries past J2000 (array ``[N]``). Returns ``(x,y,z)`` each ``[N]``.
    """
    a = _ELEM[i, 0] + _RATE[i, 0] * T
    e = _ELEM[i, 1] + _RATE[i, 1] * T
    inc = (_ELEM[i, 2] + _RATE[i, 2] * T) * DEG
    L = _ELEM[i, 3] + _RATE[i, 3] * T
    peri = _ELEM[i, 4] + _RATE[i, 4] * T
    node = (_ELEM[i, 5] + _RATE[i, 5] * T) * DEG
    omega = (peri - _ELEM[i, 5] - _RATE[i, 5] * T) * DEG      # arg. perihelion (rad)
    M = _wrap180(L - peri) * DEG
    E = _kepler_E(M, e)
    xp = a * (np.cos(E) - e)
    yp = a * np.sqrt(1.0 - e * e) * np.sin(E)
    co, so = np.cos(omega), np.sin(omega)
    ci, si = np.cos(inc), np.sin(inc)
    cn, sn = np.cos(node), np.sin(node)
    x = (co * cn - so * sn * ci) * xp + (-so * cn - co * sn * ci) * yp
    y = (co * sn + so * cn * ci) * xp + (-so * sn + co * cn * ci) * yp
    z = (so * si) * xp + (co * si) * yp
    return x, y, z


def _moon_geocentric(d):
    """Low-precision geocentric ecliptic direction of the Moon (unit vector)."""
    Lp = (218.3164477 + 13.17639648 * d) * DEG
    D = (297.8501921 + 12.19074920 * d) * DEG
    M = (357.5291092 + 0.98560028 * d) * DEG
    Mp = (134.9633964 + 13.06499295 * d) * DEG
    F = (93.2720950 + 13.22935024 * d) * DEG
    lon = (Lp / DEG
           + 6.289 * np.sin(Mp) + 1.274 * np.sin(2 * D - Mp) + 0.658 * np.sin(2 * D)
           + 0.214 * np.sin(2 * Mp) - 0.186 * np.sin(M) - 0.114 * np.sin(2 * F)
           + 0.059 * np.sin(2 * D - 2 * Mp) + 0.057 * np.sin(2 * D - M - Mp)) * DEG
    lat = (5.128 * np.sin(F) + 0.280 * np.sin(Mp + F) + 0.277 * np.sin(Mp - F)
           + 0.173 * np.sin(2 * D - F) + 0.055 * np.sin(2 * D - Mp + F)
           + 0.046 * np.sin(2 * D - Mp - F)) * DEG
    cb = np.cos(lat)
    return cb * np.cos(lon), cb * np.sin(lon), np.sin(lat)


def _node_geocentric(d):
    """Direction of the Moon's mean ascending node (on the ecliptic, beta=0)."""
    lon = (125.04452 - 0.05295377 * d) * DEG
    return np.cos(lon), np.sin(lon), np.zeros_like(lon)


def _ecl_dirs(jd):
    """Geocentric ecliptic **unit** directions of the 11 bodies at ``jd`` ([N]).

    Returns ``dirs`` of shape ``[N, 11, 3]`` (Sun, Moon, Mercury..Pluto, TrueNode).
    """
    jd = np.asarray(jd, dtype=np.float64)
    T = (jd - J2000) / 36525.0
    d = jd - J2000
    n = jd.shape[0] if jd.ndim else 1
    out = np.zeros((n, N_PLANETS_TOK, 3), dtype=np.float64)

    ex, ey, ez = _heliocentric(T, _EARTH)                    # Earth heliocentric
    # Sun (geocentric = -Earth)
    out[:, 0, :] = np.stack([-ex, -ey, -ez], axis=-1)
    # Moon
    out[:, 1, :] = np.stack(_moon_geocentric(d), axis=-1)
    # planets Mercury..Pluto (skip Earth) -> bodies index 2..9
    bi = 2
    for i in range(len(_PLANET_NAMES)):
        if i == _EARTH:
            continue
        hx, hy, hz = _heliocentric(T, i)
        out[:, bi, :] = np.stack([hx - ex, hy - ey, hz - ez], axis=-1)
        bi += 1
    # True Node (index 10)
    out[:, 10, :] = np.stack(_node_geocentric(d), axis=-1)
    # normalise to unit directions
    norm = np.linalg.norm(out, axis=-1, keepdims=True)
    return out / np.maximum(norm, 1e-12)


def _obliquity(T):
    return (23.439291 - 0.0130042 * T) * DEG                  # mean obliquity (rad)


def gmst_deg(jd):
    """Greenwich Mean Sidereal Time (degrees) — closed-form polynomial."""
    jd = np.asarray(jd, dtype=np.float64)
    T = (jd - J2000) / 36525.0
    g = (280.46061837 + 360.98564736629 * (jd - J2000)
         + 0.000387933 * T * T - T * T * T / 38710000.0)
    return g % 360.0


def asc_mc_ecliptic(lat_rad, lst_rad, ce, se):
    """Ascendant & Midheaven ecliptic longitudes (rad) for observer lat (rad), LST=RAMC (rad),
    and cos/sin obliquity. Verified against pyswisseph ``swe.houses`` to < 0.01 deg."""
    sR, cR = np.sin(lst_rad), np.cos(lst_rad)
    lam_mc = np.arctan2(sR, cR * ce)                              # MC: ecliptic pt on the meridian
    lam_asc = np.arctan2(cR, -(sR * ce + np.tan(lat_rad) * se))   # ASC: ecliptic pt rising in the E
    return lam_asc, lam_mc


def topocentric_tensor(lat_deg, lon_deg, jd):
    """The 39-D local-sky tensor for a batch of observers/times.

    ``lat_deg, lon_deg, jd`` are broadcastable arrays ``[N]``. Returns ``[N, 39]``: for each of
    the 13 tokens (11 bodies + Ascendant + Midheaven) its ``(North, East, Up)`` unit vector in
    the observer's horizontal frame. Latitude is clamped to +/-89.99 deg (pole stability, and to
    keep the ASC's ``tan(lat)`` finite) — matching the WebGL field shader exactly.
    """
    lat_deg = np.clip(np.asarray(lat_deg, dtype=np.float64), -_LAT_CLAMP, _LAT_CLAMP)
    lat = lat_deg * DEG
    lon = np.asarray(lon_deg, dtype=np.float64) * DEG
    jd = np.asarray(jd, dtype=np.float64)
    T = (jd - J2000) / 36525.0
    eps = _obliquity(T)                                      # [N]
    ce, se = np.cos(eps), np.sin(eps)
    lst = gmst_deg(jd) * DEG + lon                           # local sidereal time = RAMC [N]

    planet_dirs = _ecl_dirs(jd)                              # [N,11,3] geocentric ecliptic
    lam_asc, lam_mc = asc_mc_ecliptic(lat, lst, ce, se)      # observer-dependent [N]
    zero = np.zeros_like(lam_asc)
    asc = np.stack([np.cos(lam_asc), np.sin(lam_asc), zero], axis=-1)  # ecliptic-plane unit vecs
    mc = np.stack([np.cos(lam_mc), np.sin(lam_mc), zero], axis=-1)
    dirs = np.concatenate([planet_dirs, asc[:, None, :], mc[:, None, :]], axis=1)   # [N,13,3]

    ce2, se2 = ce[:, None], se[:, None]
    xe, ye, ze = dirs[:, :, 0], dirs[:, :, 1], dirs[:, :, 2]
    xq = xe
    yq = ye * ce2 - ze * se2                                 # ecliptic -> equatorial (rot X by eps)
    zq = ye * se2 + ze * ce2
    ra = np.arctan2(yq, xq)                                  # [N,13]
    dec = np.arcsin(np.clip(zq, -1.0, 1.0))

    H = lst[:, None] - ra                                    # hour angle [N,13]
    sphi, cphi = np.sin(lat)[:, None], np.cos(lat)[:, None]
    sd, cd = np.sin(dec), np.cos(dec)
    sH, cH = np.sin(H), np.cos(H)
    up = sd * sphi + cd * cphi * cH                          # = sin(altitude)
    north = sd * cphi - cd * sphi * cH
    east = -cd * sH
    vec = np.stack([north, east, up], axis=-1)              # [N,13,3] unit vectors
    return vec.reshape(vec.shape[0], STATE_DIM).astype(np.float32)
