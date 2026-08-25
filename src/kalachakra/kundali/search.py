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

    # -- active-set resolution (Planetary Tethers / Selective House Locking) --
    def resolve_active(self, active_planets=None, active_houses=None) -> list[str]:
        """Ordered list of the bodies whose constraints are enforced this query.

        By Houses mode (``active_houses`` given) selects the natal residents of the
        chosen houses; otherwise ``active_planets`` (default: all nine) is used.
        The order follows the canonical body order for stable SQL/scoring.
        """
        if active_houses:
            hs = {int(h) for h in active_houses}
            names = [n for n, b in self.natal["bodies"].items()
                     if b["house"] in hs]
        elif active_planets is None:
            names = list(astro.BODY_NAMES)
        else:
            want = set(active_planets)
            names = [n for n in astro.BODY_NAMES if n in want]
        return names

    # -- WHERE builders (all take the active subset ``names``) ---------------
    def _signs_where(self, names) -> str:
        b = self.natal["bodies"]
        clauses = [f"{n}_sign = {b[n]['sign']}" for n in names]
        return " AND ".join(clauses) if clauses else "TRUE"

    def _mirror_where(self, names) -> str:
        """Active bodies share a common sign offset relative to the natal chart.

        With a subset this relaxes the all-9 house-sequence lock to only the
        checked bodies (PRD Tier-3 relaxation). One or zero bodies impose no
        relative constraint (a single planet's house is always solvable).
        """
        b = self.natal["bodies"]
        if len(names) < 2:
            return "TRUE"
        ref_name = names[0]
        ref = f"(({ref_name}_sign - {b[ref_name]['sign']} + 12) % 12)"
        parts = [f"((({n}_sign - {b[n]['sign']} + 12) % 12) = {ref})"
                 for n in names[1:]]
        return " AND ".join(parts)

    def _war_where(self, names) -> str:
        active = set(names)
        clauses = []
        for group in self.natal["conjunctions"].values():   # degree-ordered
            g = [n for n in group if n in active]
            for a, c in zip(g, g[1:]):
                clauses.append(f"{a}_deg < {c}_deg")
        return " AND ".join(clauses) if clauses else "TRUE"

    def _orb_where(self, names) -> str:
        b = self.natal["bodies"]
        clauses = [f"{_circ(n + '_lon', b[n]['lon'])} <= {_ORB_DEG}" for n in names]
        return " AND ".join(clauses) if clauses else "TRUE"

    def _nav_where(self, names) -> str:
        b = self.natal["bodies"]
        clauses = [f"{n}_nav = {b[n]['nav']}" for n in names]
        return " AND ".join(clauses) if clauses else "TRUE"

    def _tier_where(self, tier: int, names) -> str:
        b = self.natal["bodies"]
        slow = [n for n in names if n in astro.SLOW_BODIES]
        if tier == 1:
            return self._signs_where(slow)
        if tier == 2:
            return self._signs_where(names)
        if tier == 3:
            return self._mirror_where(names)
        if tier == 4:
            return self._signs_where(names)
        if tier == 5:
            base = self._signs_where(names)
            return f"{base} AND moon_nak = {b['moon']['nak']}" if "moon" in names else base
        if tier == 6:
            base = self._signs_where(names)
            if "moon" in names:
                base = f"{base} AND moon_nak = {b['moon']['nak']}"
            war = self._war_where(names)
            return f"{base} AND {war}" if war != "TRUE" else base
        if tier == 7:
            return self._orb_where(names)
        if tier == 8:
            return f"({self._nav_where(names)}) AND ({self._orb_where(names)})"
        raise ValueError(f"tier must be 1..8, got {tier}")

    # -- per-row enrichment: how many of ALL nine also satisfy the criterion --
    def _total_matched(self, tier: int, row: dict) -> int:
        b = self.natal["bodies"]
        if tier in (7, 8):
            c = 0
            for n in astro.BODY_NAMES:
                d = abs(row[f"{n}_lon"] - b[n]["lon"]) % 360.0
                if min(d, 360.0 - d) <= _ORB_DEG:
                    c += 1
            return c
        if tier == 3:
            # bodies sharing the modal common sign offset k (the relative pattern)
            offs = [(int(row[f"{n}_sign"]) - b[n]["sign"]) % 12 for n in astro.BODY_NAMES]
            kmode = max(set(offs), key=offs.count)
            return sum(1 for o in offs if o == kmode)
        # sign tiers
        return sum(1 for n in astro.BODY_NAMES if int(row[f"{n}_sign"]) == b[n]["sign"])

    # -- query ---------------------------------------------------------------
    def search(self, tier: int, limit: int = 300, active_planets=None,
               active_houses=None) -> dict:
        if self.natal is None:
            raise RuntimeError("call set_natal() first")
        names = self.resolve_active(active_planets, active_houses)
        if not names:
            return {"results": [], "active_planets": [], "active_constraint_count": 0,
                    "match_score": 0.0, "note": "no active constraints"}
        where = self._tier_where(tier, names)
        # fetch jd + all nine sign/lon columns (needed for enrichment + mirror k)
        cols = ["jd"] + [f"{n}_sign" for n in astro.BODY_NAMES] \
            + [f"{n}_lon" for n in astro.BODY_NAMES]
        sql = (f"SELECT {', '.join(cols)} FROM positions WHERE {where} "
               f"ORDER BY jd LIMIT {limit}")
        raw = self.con.execute(sql).fetchall()
        natal_asc = self.natal["ascendant_sign"]

        results = []
        for tup in raw:
            row = dict(zip(cols, tup))
            jd = float(row["jd"])
            r = {"jd": jd, "date": _fmt_date(jd), "year": jd_to_gregorian(jd)[0],
                 "latitude": round(float(self.natal["lat"]), 4),
                 "total_matched": self._total_matched(tier, row)}
            if tier in _HOUSE_TIERS:
                # k is the common offset shared by the *active* bodies (tier 3);
                # tiers 4/8 match signs exactly, so k == 0.
                if tier == 3:
                    ref = names[0]
                    k = (int(row[f"{ref}_sign"]) - self.natal["bodies"][ref]["sign"]) % 12
                else:
                    k = 0
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
                r["longitude"] = round(float(self.natal["lon"]), 4)
                r["longitude_constrained"] = False
            results.append(r)
        return {
            "results": results,
            "active_planets": names,
            "active_constraint_count": len(names),
            "match_score": round(len(names) / len(astro.BODY_NAMES), 3),
        }

    def counts_by_tier(self, active_planets=None, active_houses=None) -> dict[int, bool]:
        """Cheap existence probe per tier for the current active set."""
        names = self.resolve_active(active_planets, active_houses)
        out = {}
        for t in range(1, 9):
            if not names:
                out[t] = False
                continue
            where = self._tier_where(t, names)
            row = self.con.execute(
                f"SELECT 1 FROM positions WHERE {where} LIMIT 1").fetchone()
            out[t] = row is not None
        return out
