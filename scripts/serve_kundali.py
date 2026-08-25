#!/usr/bin/env python3
"""
Kundali Twin Temporal Search Engine — standalone dashboard server.

A minimalist FastAPI app over the daily sidereal DuckDB (built by
scripts/build_kundali_db.py). It is completely independent of the neural network
and the 3-D archetype mesh: the user submits a birth instant + place, drags a
slider across eight escalating similarity tiers, and the backend runs one indexed
SQL sweep per tier to return the historical days (and, for house tiers, the
geographic longitudes) where their astrological twin could have existed.

Endpoints:
  GET  /                       -> the 2-D map dashboard (web/kundali.html)
  GET  /health                 -> db coverage summary
  GET  /api/coastlines.geojson -> world outline for the 2-D map
  POST /api/search             -> {date,time,tz_hours,lat,lon,tier,limit} ->
                                  natal chart + tier hits + per-tier availability

Requires:  pip install "kalachakra[index,serve]"  (duckdb + fastapi/uvicorn) and a
built DuckDB (scripts/build_kundali_db.py).

Example:
    python scripts/build_kundali_db.py --start-year 1800 --end-year 2100 --out kundali.duckdb
    python scripts/serve_kundali.py --db kundali.duckdb
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

_ROOT = Path(__file__).resolve().parents[1]
_WEB = _ROOT / "web" / "kundali.html"
_COAST = _ROOT / "web" / "coastlines.geojson"


def _natal_jd(date: str, time: str, tz_hours: float) -> float:
    """(YYYY-MM-DD, HH:MM, tz offset from UT) -> Julian Day (UT)."""
    from kalachakra.ephemeris.calendar import gregorian_to_jd

    y, m, d = (int(x) for x in date.split("-"))
    hh, mm = (int(x) for x in time.split(":")[:2]) if ":" in time else (int(time), 0)
    local_jd = gregorian_to_jd(y, m, d, hh, mm)
    return local_jd - tz_hours / 24.0


def _natal_summary(natal: dict) -> dict:
    from kalachakra.kundali import astro
    return {
        "ascendant": astro.SIGNS[natal["ascendant_sign"]],
        "bodies": {n: {"sign": astro.SIGNS[b["sign"]], "deg": round(b["deg"], 2),
                       "nakshatra": astro.NAKSHATRAS[b["nak"]],
                       "navamsa": astro.SIGNS[b["nav"]]}
                   for n, b in natal["bodies"].items()},
    }


def create_app(db_path: str):
    from fastapi import Body, FastAPI, Response
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, JSONResponse

    from kalachakra.kundali import astro
    from kalachakra.kundali.search import TIER_NAMES, KundaliSearch

    if not Path(db_path).exists():
        raise FileNotFoundError(f"kundali DuckDB not found: {db_path} "
                                "(build it with scripts/build_kundali_db.py)")
    # read coverage once for /health
    import duckdb
    con = duckdb.connect(db_path, read_only=True)
    meta = dict(con.execute("SELECT key, value FROM meta").fetchall())
    n_days = con.execute("SELECT COUNT(*) FROM positions").fetchone()[0]
    con.close()

    app = FastAPI(title="Kundali Twin Temporal Search Engine", version="1.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                       allow_headers=["*"])

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "n_days": int(n_days),
                "start_year": meta.get("start_year"), "end_year": meta.get("end_year"),
                "ayanamsha": meta.get("ayanamsha", "lahiri"),
                "tiers": TIER_NAMES}

    @app.get("/api/coastlines.geojson")
    def coastlines():
        if _COAST.is_file():
            return FileResponse(str(_COAST), media_type="application/geo+json")
        return Response("{}", media_type="application/json")

    @app.post("/api/search")
    def search(body: dict = Body(...)) -> JSONResponse:
        try:
            jd = _natal_jd(body["date"], body.get("time", "12:00"),
                           float(body.get("tz_hours", 0.0)))
            lat = float(body["lat"])
            lon = float(body["lon"])
            tier = int(body.get("tier", 1))
            limit = int(body.get("limit", 300))
        except (KeyError, ValueError) as exc:
            return JSONResponse({"error": f"bad request: {exc}"}, status_code=400)

        ks = KundaliSearch(db_path)
        try:
            natal = ks.set_natal(jd, lat, lon)
            results = ks.search(tier, limit=limit)
            available = ks.counts_by_tier()
        finally:
            ks.close()
        return JSONResponse({
            "tier": tier, "tier_name": TIER_NAMES[tier], "count": len(results),
            "results": results, "available": {str(k): v for k, v in available.items()},
            "tier_names": TIER_NAMES,
            "natal": _natal_summary(natal),
            "signs": list(astro.SIGNS),
        })

    @app.get("/")
    def index():
        if _WEB.is_file():
            return FileResponse(str(_WEB))
        return Response("web/kundali.html not found", status_code=404)

    return app


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default="kundali.duckdb", help="path to the sidereal DuckDB")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not Path(args.db).exists():
        print(f"ERROR: db not found: {args.db}\nBuild it first:\n"
              f"  python scripts/build_kundali_db.py --start-year 1800 --end-year 2100 "
              f"--out {args.db}", file=sys.stderr)
        return 2
    try:
        import uvicorn
    except Exception:
        print('ERROR: uvicorn not installed. `pip install "kalachakra[serve]"`.',
              file=sys.stderr)
        return 2
    app = create_app(args.db)
    print(f"\nKundali Twin engine on http://{args.host}:{args.port}  (open in a browser)")
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
