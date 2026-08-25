"""Sidereal (Vedic) astrology math for the Kundali Twin search engine.

Relies strictly on raw Swiss Ephemeris data (no neural network, no mesh). All
longitudes are **sidereal** using the Lahiri ayanamsha, which is the standard for
Vedic (Jyotish) charts. Pure/deterministic helpers here so the DB builder, the
search engine and the tests all share one source of truth.

Zodiac signs are 0..11 (Aries..Pisces); nakshatras 0..26; navamsa (D9) signs
0..11. Degrees are within-sign [0, 30).
"""

from __future__ import annotations

import numpy as np

from .. import geometry as geo
from ..ephemeris import global_state as gs
from ..projection.spatial import ascendant_longitude

SIGNS = ("Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
         "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces")
NAKSHATRAS = (
    "Ashwini", "Bharani", "Krittika", "Rohini", "Mrigashira", "Ardra", "Punarvasu",
    "Pushya", "Ashlesha", "Magha", "PurvaPhalguni", "UttaraPhalguni", "Hasta",
    "Chitra", "Swati", "Vishakha", "Anuradha", "Jyeshtha", "Mula", "PurvaAshadha",
    "UttaraAshadha", "Shravana", "Dhanishta", "Shatabhisha", "PurvaBhadrapada",
    "UttaraBhadrapada", "Revati")

#: The nine Jyotish grahas in the PRD's canonical order, with Swiss Ephemeris ids.
#: Rahu is the true north node; Ketu is exactly 180 deg opposite.
BODIES: tuple[tuple[str, int, float], ...] = (
    ("sun", 0, 0.0), ("moon", 1, 0.0), ("mars", 4, 0.0), ("mercury", 2, 0.0),
    ("jupiter", 5, 0.0), ("venus", 3, 0.0), ("saturn", 6, 0.0),
    ("rahu", 11, 0.0), ("ketu", 11, 180.0),
)
BODY_NAMES = tuple(b[0] for b in BODIES)
#: Slowest-moving bodies (Tier 1 "Generational Twin").
SLOW_BODIES = ("saturn", "jupiter", "rahu", "ketu")

_NAV_ARC = 30.0 / 9.0           # navamsa arc = 3 deg 20 min
_NAK_ARC = 360.0 / 27.0        # nakshatra arc = 13 deg 20 min

_CONFIGURED = False


def configure() -> None:
    """Ensure Swiss Ephemeris is available and set to the Lahiri sidereal zodiac."""
    global _CONFIGURED
    gs._require_swe()
    if not _CONFIGURED:
        gs.auto_configure()
        gs.swe.set_sid_mode(gs.swe.SIDM_LAHIRI, 0, 0)
        _CONFIGURED = True


def sign_of(lon_deg: float) -> int:
    return int(lon_deg % 360.0 // 30.0)


def degree_in_sign(lon_deg: float) -> float:
    return float(lon_deg % 30.0)


def nakshatra_of(lon_deg: float) -> int:
    return int(lon_deg % 360.0 // _NAK_ARC)


def navamsa_sign(lon_deg: float) -> int:
    """D9 navamsa sign (0..11) using the standard element-based mapping.

    Equivalent closed form: ``(sign*9 + navamsa_index) % 12`` where the navamsa
    index is the 3deg20' slice within the sign. This reproduces the movable/fixed/
    dual start rule (movable -> same sign, fixed -> 9th, dual -> 5th).
    """
    sign = sign_of(lon_deg)
    idx = int(degree_in_sign(lon_deg) // _NAV_ARC)
    return (sign * 9 + idx) % 12


def body_longitude(jd_ut: float, swe_id: int, offset_deg: float = 0.0) -> float:
    """Sidereal ecliptic longitude (deg, [0,360)) of one body at ``jd_ut``."""
    flags = gs._calc_flags() | gs.swe.FLG_SIDEREAL
    values, _flag = gs.swe.calc_ut(jd_ut, swe_id, flags)
    return float((values[0] + offset_deg) % 360.0)


def body_longitudes(jd_ut: float) -> dict[str, float]:
    """Sidereal longitudes (deg) of all nine grahas at ``jd_ut``."""
    return {name: body_longitude(jd_ut, sid, off) for name, sid, off in BODIES}


def ayanamsha(jd_ut: float) -> float:
    return float(gs.swe.get_ayanamsa_ut(jd_ut))


def ascendant_sign(jd_ut: float, geo_lat_deg: float, geo_lon_deg: float) -> int:
    """Sidereal rising-sign (Lagna) at a place and instant."""
    return sign_of(ascendant_sidereal_deg(jd_ut, geo_lat_deg, geo_lon_deg))


def ascendant_sidereal_deg(jd_ut: float, geo_lat_deg: float, geo_lon_deg: float) -> float:
    """Sidereal ecliptic longitude (deg) of the Ascendant."""
    gmst = float(geo.greenwich_mean_sidereal_time_deg(jd_ut))
    ramc = np.deg2rad((gmst + geo_lon_deg) % 360.0)
    eps = float(geo.obliquity_of_ecliptic(jd_ut))
    asc_trop = float(np.rad2deg(ascendant_longitude(ramc, eps, np.deg2rad(geo_lat_deg))))
    return (asc_trop - ayanamsha(jd_ut)) % 360.0


def ascendant_signs_over_longitudes(jd_ut: float, geo_lat_deg: float,
                                    lons_deg: np.ndarray) -> np.ndarray:
    """Vectorized sidereal Ascendant sign (0..11) for many geographic longitudes.

    The obliquity, GMST and ayanamsha are constant for the instant, so they are
    computed once and only the RAMC (= GMST + longitude) varies — this makes
    solving 1000+ dates (and a latitude spread) stay well under a second.
    """
    gmst = float(geo.greenwich_mean_sidereal_time_deg(jd_ut))
    eps = float(geo.obliquity_of_ecliptic(jd_ut))
    ayan = ayanamsha(jd_ut)
    ramc = np.deg2rad((gmst + np.asarray(lons_deg, dtype=np.float64)) % 360.0)
    asc_trop = np.rad2deg(ascendant_longitude(ramc, eps, np.deg2rad(geo_lat_deg)))
    asc_sid = (asc_trop - ayan) % 360.0
    return (asc_sid // 30.0).astype(np.int64)


def _bands_from_hit(lons: np.ndarray, hit: np.ndarray) -> list[tuple[int, float]]:
    """Contiguous ``True`` runs of ``hit`` over the circular ``lons`` grid.

    Returns ``(width, center_longitude)`` pairs sorted widest-first, with the
    center taken as the circular mean so a band straddling the +/-180 seam is
    handled correctly.
    """
    bands: list[tuple[int, float]] = []
    n = len(lons)
    start = None
    for i in range(n + 1):
        on = bool(hit[i % n]) if i < n else False
        if on and start is None:
            start = i
        elif not on and start is not None:
            seg = [lons[j % n] for j in range(start, i)]
            center = float(np.mean(np.unwrap(np.deg2rad(seg))))
            center = (np.rad2deg(center) + 180.0) % 360.0 - 180.0
            bands.append((len(seg), center))
            start = None
    bands.sort(key=lambda b: -b[0])
    return bands


def longitudes_for_ascendant_sign(jd_ut: float, geo_lat_deg: float, target_sign: int,
                                  coarse_step: float = 0.5, refine: bool = False):
    """Geographic longitudes (deg, -180..180) that put the Lagna in ``target_sign``.

    Over a day the Ascendant cycles through all twelve signs as the Earth turns, so
    for essentially any date there is a band of longitudes yielding the target
    rising sign at the given latitude. Returns the center longitude of each
    contiguous band (widest first). ``refine`` narrows the top center toward
    arcminute precision for the divisional-lock tier.
    """
    lons = np.arange(-180.0, 180.0, coarse_step)
    hit = ascendant_signs_over_longitudes(jd_ut, geo_lat_deg, lons) == target_sign
    if not hit.any():
        return []
    bands = _bands_from_hit(lons, hit)
    centers = [round(c, 4) for _w, c in bands]
    if refine and centers:
        centers[0] = _refine_center(jd_ut, geo_lat_deg, target_sign, centers[0])
    return centers


#: Default latitude fan for the twin locus. The Ascendant depends on the
#: observer's latitude, so a fixed target rising-sign traces a lat/lon curve
#: across the globe rather than a single point.
GLOBE_LATS: tuple[float, ...] = (-55.0, -40.0, -25.0, -10.0, 0.0,
                                 10.0, 25.0, 40.0, 55.0)


def globe_ascendant_points(jd_ut: float, target_sign: int,
                           lats_deg=GLOBE_LATS, coarse_step: float = 0.5):
    """The twin locus: ``(lat, lon)`` points across latitudes for one rising sign.

    For each latitude in ``lats_deg`` this returns the widest longitude band-center
    that puts the Lagna in ``target_sign`` at ``jd_ut`` — the set of places on Earth
    where the natal house pattern is reproduced at that instant. The obliquity,
    GMST and ayanamsha are constant for the instant, so the whole (lat x lon) grid
    is evaluated in one broadcast for speed.
    """
    gmst = float(geo.greenwich_mean_sidereal_time_deg(jd_ut))
    eps = float(geo.obliquity_of_ecliptic(jd_ut))
    ayan = ayanamsha(jd_ut)
    lons = np.arange(-180.0, 180.0, coarse_step)
    lats = np.asarray(lats_deg, dtype=np.float64)
    ramc = np.deg2rad((gmst + lons) % 360.0)                       # (L,)
    asc_trop = np.rad2deg(ascendant_longitude(
        ramc[None, :], eps, np.deg2rad(lats)[:, None]))           # (K, L)
    signs = ((asc_trop - ayan) % 360.0 // 30.0).astype(np.int64)  # (K, L)
    points = []
    for i in range(lats.size):
        bands = _bands_from_hit(lons, signs[i] == target_sign)
        if bands:
            points.append({"lat": round(float(lats[i]), 2),
                           "lon": round(bands[0][1], 3)})
    return points


def _refine_center(jd_ut, geo_lat_deg, target_sign, approx_lon, span=1.0):
    """Refine an ascendant-sign band center toward arcminute longitude precision."""
    lons = np.arange(approx_lon - span, approx_lon + span, 1.0 / 60.0)
    hit = np.nonzero(ascendant_signs_over_longitudes(jd_ut, geo_lat_deg, lons)
                     == target_sign)[0]
    if hit.size == 0:
        return round(float(approx_lon), 5)
    return round(float(lons[hit[hit.size // 2]]), 5)


def natal_chart(jd_ut: float, geo_lat_deg: float, geo_lon_deg: float) -> dict:
    """Full sidereal natal chart used as the search key.

    Returns per-body ``lon/sign/deg/nak/nav`` plus the Ascendant sign and the
    conjunction groups (bodies sharing a sign, with their degree order) needed by
    the higher search tiers.
    """
    lons = body_longitudes(jd_ut)
    asc = ascendant_sign(jd_ut, geo_lat_deg, geo_lon_deg)
    bodies = {}
    houses: dict[int, list[str]] = {}
    for name in BODY_NAMES:
        lo = lons[name]
        sign = sign_of(lo)
        house = (sign - asc) % 12 + 1                # whole-sign house (1..12)
        bodies[name] = {
            "lon": round(lo, 5), "sign": sign, "deg": round(degree_in_sign(lo), 5),
            "nak": nakshatra_of(lo), "nav": navamsa_sign(lo), "house": house,
        }
        houses.setdefault(house, []).append(name)
    # conjunction groups: bodies sharing a sign, ordered by degree (who "wins")
    by_sign: dict[int, list[str]] = {}
    for name in BODY_NAMES:
        by_sign.setdefault(bodies[name]["sign"], []).append(name)
    conj = {s: sorted(names, key=lambda nm: bodies[nm]["deg"])
            for s, names in by_sign.items() if len(names) > 1}
    return {
        "jd": jd_ut, "lat": geo_lat_deg, "lon": geo_lon_deg,
        "ascendant_sign": asc, "bodies": bodies, "conjunctions": conj,
        "houses": {h: sorted(names) for h, names in sorted(houses.items())},
    }
