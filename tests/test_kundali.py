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
    prev_broad = None
    for tier in range(1, 9):
        res = ks.search(tier, limit=500)
        for r in res:
            assert "jd" in r and "date" in r and "longitude" in r
            if tier in house_tiers:
                assert r["longitude_constrained"] is True
                assert "ascendant_sign" in r and 0 <= r["ascendant_sign"] <= 11
            else:
                assert r["longitude_constrained"] is False
        if tier == 2:                     # the birth day itself is a psychological twin
            assert any(r["date"].startswith("1990-01-01") for r in res)
        prev_broad = res
    assert prev_broad is not None
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
    res = ks.search(7, limit=50)
    for r in res:
        lons = astro.body_longitudes(r["jd"])
        for name, b in natal["bodies"].items():
            d = abs(lons[name] - b["lon"]) % 360
            d = min(d, 360 - d)
            assert d <= 5.0 + 1e-6
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
    assert "ascendant" in r["natal"] and len(r["natal"]["bodies"]) == 9
    assert set(r["available"]) == {str(i) for i in range(1, 9)}

    # bad request -> 400
    assert c.post("/api/search", json={"date": "1990-01-01"}).status_code == 400
    # coastlines route serves geojson
    assert c.get("/api/coastlines.geojson").status_code == 200
