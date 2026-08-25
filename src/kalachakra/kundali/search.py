"""The eight-tier Kundali Twin search over the daily sidereal DuckDB.

Given a natal chart (birth instant + place), each tier is one indexed SQL sweep
that returns the historical days matching at an escalating level of astrological
rarity, plus — for the house/ascendant tiers — the geographic longitude(s) that
would place the Lagna correctly. Empty results are returned as an empty list so
the UI can invite the user to a broader tier.

Tiers (broad -> rare):
  1 Generational Twin   slow bodies (Saturn/Jupiter/Rahu/Ketu) share signs
  2 Psychological Twin  all nine bodies share signs (houses ignored)
  3 Geographic Mirror   same 1..12 house sequence (signs free); solve longitude
  4 Core Ascendant Lock all signs match AND Lagna locked to the natal sign
  5 Nakshatra Twin      tier 4 + Moon in the same nakshatra
  6 Planetary War Lock  tier 5 + conjunct planets keep the same degree order
  7 Absolute Twin       every body within a 5deg orb of the natal degree
  8 Divisional Lock     D9 navamsa identical + tier-7 orb + refined longitude
"""

from __future__ import annotations

from ..ephemeris.calendar import jd_to_gregorian
from . import astro

TIER_NAMES = {
    1: "Generational Twin", 2: "Psychological Twin", 3: "Geographic Mirror",
    4: "Core Ascendant Lock", 5: "Nakshatra Twin", 6: "Planetary War Lock",
    7: "Absolute Twin", 8: "Divisional Lock",
}
_HOUSE_TIERS = {3, 4, 8}
_ORB_DEG = 5.0


def _fmt_date(jd: float) -> str:
    y, m, d, *_ = jd_to_gregorian(float(jd))
    era = "CE" if y > 0 else "BCE"
    yy = y if y > 0 else 1 - y
    return f"{yy:04d}-{m:02d}-{d:02d} {era}"


def _circ(col: str, val: float) -> str:
    return f"LEAST(ABS({col} - {val}), 360 - ABS({col} - {val}))"


class KundaliSearch:
    """Loads the sidereal DuckDB and answers tiered twin queries for one natal chart."""

    def __init__(self, db_path: str):
        import duckdb
        self.con = duckdb.connect(str(db_path), read_only=True)
        self.natal: dict | None = None

    def close(self):
        self.con.close()

    def set_natal(self, jd_ut: float, lat_deg: float, lon_deg: float) -> dict:
        astro.configure()
        self.natal = astro.natal_chart(jd_ut, lat_deg, lon_deg)
        return self.natal

    # -- WHERE builders ------------------------------------------------------
    def _signs_where(self, names) -> str:
        b = self.natal["bodies"]
        return " AND ".join(f"{n}_sign = {b[n]['sign']}" for n in names)

    def _mirror_where(self) -> str:
        """All bodies shifted by a common sign offset relative to the natal chart."""
        b = self.natal["bodies"]
        ref = f"((sun_sign - {b['sun']['sign']} + 12) % 12)"
        parts = [f"((({n}_sign - {b[n]['sign']} + 12) % 12) = {ref})"
                 for n in astro.BODY_NAMES if n != "sun"]
        return " AND ".join(parts)

    def _war_where(self) -> str:
        """Conjunct planets keep the same degree order (same planet 'wins')."""
        clauses = []
        for group in self.natal["conjunctions"].values():   # degree-ordered
            for a, c in zip(group, group[1:]):
                clauses.append(f"{a}_deg < {c}_deg")
        return " AND ".join(clauses)

    def _orb_where(self) -> str:
        b = self.natal["bodies"]
        return " AND ".join(f"{_circ(n + '_lon', b[n]['lon'])} <= {_ORB_DEG}"
                            for n in astro.BODY_NAMES)

    def _nav_where(self) -> str:
        b = self.natal["bodies"]
        return " AND ".join(f"{n}_nav = {b[n]['nav']}" for n in astro.BODY_NAMES)

    def _tier_where(self, tier: int) -> str:
        b = self.natal["bodies"]
        if tier == 1:
            return self._signs_where(astro.SLOW_BODIES)
        if tier == 2:
            return self._signs_where(astro.BODY_NAMES)
        if tier == 3:
            return self._mirror_where()
        if tier == 4:
            return self._signs_where(astro.BODY_NAMES)
        if tier == 5:
            return f"{self._signs_where(astro.BODY_NAMES)} AND moon_nak = {b['moon']['nak']}"
        if tier == 6:
            base = f"{self._signs_where(astro.BODY_NAMES)} AND moon_nak = {b['moon']['nak']}"
            war = self._war_where()
            return f"{base} AND {war}" if war else base
        if tier == 7:
            return self._orb_where()
        if tier == 8:
            return f"{self._nav_where()} AND {self._orb_where()}"
        raise ValueError(f"tier must be 1..8, got {tier}")

    # -- query ---------------------------------------------------------------
    def search(self, tier: int, limit: int = 300) -> list[dict]:
        if self.natal is None:
            raise RuntimeError("call set_natal() first")
        where = self._tier_where(tier)
        # sun_sign is needed to recover the mirror offset k per row (tier 3)
        rows = self.con.execute(
            f"SELECT jd, year, sun_sign FROM positions WHERE {where} "
            f"ORDER BY jd LIMIT {limit}").fetchall()
        natal_asc = self.natal["ascendant_sign"]
        natal_sun = self.natal["bodies"]["sun"]["sign"]

        results = []
        for jd, _year, sun_sign in rows:
            jd = float(jd)
            r = {"jd": jd, "date": _fmt_date(jd), "year": jd_to_gregorian(jd)[0],
                 "latitude": round(float(self.natal["lat"]), 4)}
            if tier in _HOUSE_TIERS:
                # tier 3 shifts the ascendant by the common sign offset k; tiers 4/8
                # match signs exactly (k == 0), so the target is the natal Lagna.
                k = (int(sun_sign) - natal_sun) % 12 if tier == 3 else 0
                target = (natal_asc + k) % 12
                bands = astro.longitudes_for_ascendant_sign(
                    jd, self.natal["lat"], target, refine=(tier == 8))
                if not bands:
                    continue
                r["longitude"] = bands[0]
                r["longitudes"] = bands
                r["ascendant_sign"] = target
                r["longitude_constrained"] = True
            else:
                # sign/degree tiers are longitude-free: the twin can be born at any
                # longitude that day; we pin at the user's own longitude as a proxy.
                r["longitude"] = round(float(self.natal["lon"]), 4)
                r["longitude_constrained"] = False
            results.append(r)
        return results

    def counts_by_tier(self, limit_probe: int = 1) -> dict[int, bool]:
        """Cheap existence probe per tier (does any match exist?)."""
        out = {}
        for t in range(1, 9):
            where = self._tier_where(t)
            row = self.con.execute(
                f"SELECT 1 FROM positions WHERE {where} LIMIT 1").fetchone()
            out[t] = row is not None
        return out
