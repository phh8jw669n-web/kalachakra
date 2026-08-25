"""Kundali Twin engine: sidereal math, the daily DuckDB, and the 8-tier search.

Builds a small sidereal ephemeris DB over a recent span (within the Moshier
backend's range), then exercises the tier queries and the dashboard API.
"""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")

from kalachakra.ephemeris import global_state                        # noqa: E402
from kalachakra.ephemeris.calendar import gregorian_to_jd, parse_datetime  # noqa: E402


def _skip_if_no_ephem():
    if not global_state.ephemeris_available():
        pytest.skip("pyswisseph not installed")


# ---------------------------------------------------------------------------
# astrology math
# ---------------------------------------------------------------------------
def test_astro_sign_nakshatra_navamsa():
    from kalachakra.kundali import astro
    assert astro.sign_of(0.0) == 0 and astro.sign_of(359.9) == 11
    assert astro.nakshatra_of(0.0) == 0 and astro.nakshatra_of(359.9) == 26
    # navamsa element rule: movable->same, fixed->9th, dual->5th
    assert astro.navamsa_sign(0.1) == 0        # Aries (movable)
    assert astro.navamsa_sign(30.1) == 9       # Taurus (fixed) -> Capricorn
    assert astro.navamsa_sign(60.1) == 6       # Gemini (dual) -> Libra
    assert astro.navamsa_sign(119.99) == 11    # last navamsa of Cancer (movable) -> Pisces
    for lon in (0.0, 45.5, 123.4, 271.9, 359.99):
        assert 0 <= astro.navamsa_sign(lon) <= 11


def test_natal_chart_and_ascendant_band():
    _skip_if_no_ephem()
    from kalachakra.kundali import astro
    astro.configure()
    jd = parse_datetime("1990-01-01T12:00:00Z")     # noon UT == integer JD
    chart = astro.natal_chart(jd, 26.9, 75.8)
    # Rahu and Ketu are always exactly opposite
    assert (chart["bodies"]["rahu"]["sign"] - chart["bodies"]["ketu"]["sign"]) % 12 == 6
    # every body has valid sign / nak / nav
    for b in chart["bodies"].values():
        assert 0 <= b["sign"] <= 11 and 0 <= b["nak"] <= 26 and 0 <= b["nav"] <= 11
    # the longitude band for the natal ascendant sign, at the birth instant,
    # includes the birth longitude
    bands = astro.longitudes_for_ascendant_sign(jd, 26.9, chart["ascendant_sign"])
    assert bands and min(abs(b - 75.8) for b in bands) < 8.0


def test_globe_locus_spans_latitudes():
    """A fixed rising sign is reproduced along a lat/lon curve, not one point."""
    _skip_if_no_ephem()
    from kalachakra.kundali import astro
    astro.configure()
    jd = parse_datetime("1990-01-01T12:00:00Z")
    pts = astro.globe_ascendant_points(jd, 3)
    assert len(pts) >= 5
    assert len({p["lat"] for p in pts}) > 1                 # spans latitudes
    # every locus point actually rises the target sign at its own latitude
    for p in pts:
        assert astro.ascendant_sign(jd, p["lat"], p["lon"]) == 3


# ---------------------------------------------------------------------------
# DB + search
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def kundali_db(tmp_path_factory):
    _skip_if_no_ephem()
    from kalachakra.kundali import db
    path = tmp_path_factory.mktemp("kundali") / "k.duckdb"
    s = gregorian_to_jd(1985, 1, 1)
    e = gregorian_to_jd(1995, 1, 1)
    db.build_db(str(path), s, e)
    return str(path)


def test_db_schema_and_ranges(kundali_db):
    import duckdb
    from kalachakra.kundali import astro
    con = duckdb.connect(kundali_db, read_only=True)
    cols = {r[1] for r in con.execute("PRAGMA table_info(positions)").fetchall()}
    for b in astro.BODY_NAMES:
        for suf in ("sign", "deg", "nak", "nav", "lon"):
            assert f"{b}_sign" in cols and f"{b}_{suf}" in cols
    n = con.execute("SELECT COUNT(*) FROM positions").fetchone()[0]
    assert n > 3000                       # ~10 years of daily rows
    lo, hi = con.execute("SELECT MIN(sun_sign), MAX(sun_sign) FROM positions").fetchone()
    assert lo >= 0 and hi <= 11
    con.close()


def test_all_tiers_and_structure(kundali_db):
    from kalachakra.kundali.search import KundaliSearch
    ks = KundaliSearch(kundali_db)
    ks.set_natal(parse_datetime("1990-01-01T12:00:00Z"), 26.9, 75.8)
    house_tiers = {3, 4, 8}
    for tier in range(1, 9):
        out = ks.search(tier, limit=500)
        assert out["active_constraint_count"] == 9 and out["match_score"] == 1.0
        assert out["location_free"] is (tier not in house_tiers)
        res = out["results"]
        for r in res:
            assert "jd" in r and "date" in r and "longitude" in r
            assert "time_utc" in r and r["time_utc"].endswith("UTC")
            assert 1 <= r["total_matched"] <= 9
            if tier in house_tiers:
                assert r["longitude_constrained"] is True
                assert "ascendant_sign" in r and 0 <= r["ascendant_sign"] <= 11
                assert "local_time" in r
            else:
                assert r["longitude_constrained"] is False
        if tier in house_tiers and res:
            # the leading result carries the globe locus, and it spans latitudes
            gp = res[0].get("globe_points")
            assert gp and len({p["lat"] for p in gp}) > 1
        if tier == 2:                     # the birth day itself is a psychological twin
            assert any(r["date"].startswith("1990-01-01") for r in res)
    avail = ks.counts_by_tier()
    assert set(avail) == set(range(1, 9)) and all(isinstance(v, bool) for v in avail.values())
    ks.close()


def test_absolute_twin_orb(kundali_db):
    # Tier 7 requires every body within 5deg of the natal longitude; on the birth
    # day (if present) the orb is ~0. Verify any returned day respects the orb.
    from kalachakra.kundali import astro
    from kalachakra.kundali.search import KundaliSearch
    ks = KundaliSearch(kundali_db)
    natal = ks.set_natal(parse_datetime("1990-01-01T12:00:00Z"), 26.9, 75.8)
    res = ks.search(7, limit=50)["results"]
    for r in res:
        lons = astro.body_longitudes(r["jd"])
        for name, b in natal["bodies"].items():
            d = abs(lons[name] - b["lon"]) % 360
            d = min(d, 360 - d)
            assert d <= 5.0 + 1e-6
    ks.close()


def test_year_range_filter(kundali_db):
    """The optional date range restricts results to the requested span."""
    from kalachakra.kundali.search import KundaliSearch
    ks = KundaliSearch(kundali_db)
    ks.set_natal(parse_datetime("1990-01-01T12:00:00Z"), 26.9, 75.8)
    everything = ks.search(1, limit=5000)["results"]
    ranged = ks.search(1, limit=5000, start_year=1990, end_year=1991)["results"]
    assert 0 < len(ranged) < len(everything)
    assert all(1990 <= r["year"] <= 1991 for r in ranged)
    # availability probe honours the range too
    avail = ks.counts_by_tier(start_year=1990, end_year=1990)
    assert set(avail) == set(range(1, 9))
    ks.close()


def test_dynamic_constraint_toggles(kundali_db):
    """Relaxing the active set widens Tier 3; the full set preserves strict counts."""
    from kalachakra.kundali import astro
    from kalachakra.kundali.search import KundaliSearch
    ks = KundaliSearch(kundali_db)
    ks.set_natal(parse_datetime("1990-01-01T12:00:00Z"), 26.9, 75.8)

    full = ks.search(3, limit=2000)                     # all 9 bodies (strict)
    relaxed = ks.search(3, limit=2000,                  # drop fast movers
                        active_planets=["sun", "jupiter", "saturn", "rahu", "ketu"])
    assert full["active_constraint_count"] == 9
    assert relaxed["active_constraint_count"] == 5
    assert relaxed["match_score"] == round(5 / 9, 3)
    # relaxing the constraint set must not shrink the match set
    assert len(relaxed["results"]) >= len(full["results"])
    # a strict all-9 query equals the default (no active set)
    assert len(ks.search(2)["results"]) == len(
        ks.search(2, active_planets=list(astro.BODY_NAMES))["results"])

    # By Houses mode: locking the houses of a couple of bodies resolves to a
    # planet subset and returns matches with solved longitudes.
    natal = ks.natal
    occ = sorted(int(h) for h in natal["houses"])       # occupied houses
    houses = occ[:2]
    byh = ks.search(3, limit=500, active_houses=houses)
    assert byh["active_constraint_count"] >= 1
    for r in byh["results"][:5]:
        assert r["longitude_constrained"] is True
    ks.close()


# ---------------------------------------------------------------------------
# dashboard API
# ---------------------------------------------------------------------------
def _load_server():
    pytest.importorskip("fastapi")
    p = Path(__file__).resolve().parents[1] / "scripts" / "serve_kundali.py"
    spec = importlib.util.spec_from_file_location("serve_kundali", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_dashboard_api(kundali_db):
    from fastapi.testclient import TestClient
    sk = _load_server()
    app = sk.create_app(kundali_db)
    c = TestClient(app)

    h = c.get("/health").json()
    assert h["status"] == "ok" and h["n_days"] > 3000

    r = c.post("/api/search", json={"date": "1990-01-01", "time": "12:00",
                                    "tz_hours": 0.0, "lat": 26.9, "lon": 75.8,
                                    "tier": 2}).json()
    assert r["tier"] == 2 and r["tier_name"] == "Psychological Twin"
    assert r["count"] >= 1
    assert r["location_free"] is True
    assert r["results"][0]["time_utc"].endswith("UTC")
    assert "coverage" in r and r["coverage"]["start_year"] <= 1990
    assert "ascendant" in r["natal"] and len(r["natal"]["bodies"]) == 9
    assert set(r["available"]) == {str(i) for i in range(1, 9)}

    # date range narrows the sweep
    rr = c.post("/api/search", json={"date": "1990-01-01", "time": "12:00",
                                     "tz_hours": 0.0, "lat": 26.9, "lon": 75.8,
                                     "tier": 1, "start_year": 1990,
                                     "end_year": 1991}).json()
    assert rr["count"] >= 1 and all(1990 <= x["year"] <= 1991 for x in rr["results"])

    # bad request -> 400
    assert c.post("/api/search", json={"date": "1990-01-01"}).status_code == 400
    # coastlines route serves geojson
    assert c.get("/api/coastlines.geojson").status_code == 200
