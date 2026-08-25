"""Builds the lightweight daily sidereal-ephemeris DuckDB for the Kundali engine.

One row per day carries the nine grahas' sidereal sign / degree / nakshatra /
navamsa (+ longitude), wide so the tier SQL can match signs across bodies in one
scan. Sign columns are indexed so the sweeps resolve in milliseconds. This is a
secondary, standalone database — entirely independent of the mesh / neural core.

The full Holocene needs a wide-range backend (DE441 / Swiss files); the Moshier
default only spans ~3000 BCE-3000 CE, so pass a range your ephemeris covers.
"""

from __future__ import annotations

import time

import numpy as np

from ..ephemeris.calendar import jd_to_gregorian
from . import astro

_BODY_COLS = ("sign", "deg", "nak", "nav", "lon")


def build_arrays(start_jd: float, end_jd: float, step_days: float = 1.0,
                 logger=None) -> dict:
    """Compute the daily columns between two Julian Days, sampled at noon UT.

    Integer Julian Days fall at 12:00 UT, so noon anchoring == integer JDs; this
    also keeps the search self-consistent when a birth given at 12:00 UT lands
    exactly on its own day's row.
    """
    astro.configure()
    base = float(np.ceil(start_jd - 1e-9))          # first noon-UT (integer JD) >= start
    jds = np.arange(base, end_jd, step_days)
    n = jds.size
    cols: dict[str, np.ndarray] = {
        "jd": jds.astype(np.float64),
        "year": np.empty(n, dtype=np.int32),
    }
    for name in astro.BODY_NAMES:
        cols[f"{name}_sign"] = np.empty(n, dtype=np.int8)
        cols[f"{name}_deg"] = np.empty(n, dtype=np.float32)
        cols[f"{name}_nak"] = np.empty(n, dtype=np.int8)
        cols[f"{name}_nav"] = np.empty(n, dtype=np.int8)
        cols[f"{name}_lon"] = np.empty(n, dtype=np.float32)

    t0 = time.time()
    for i, jd in enumerate(jds):
        cols["year"][i] = jd_to_gregorian(float(jd))[0]
        lons = astro.body_longitudes(float(jd))
        for name in astro.BODY_NAMES:
            lo = lons[name]
            cols[f"{name}_sign"][i] = astro.sign_of(lo)
            cols[f"{name}_deg"][i] = astro.degree_in_sign(lo)
            cols[f"{name}_nak"][i] = astro.nakshatra_of(lo)
            cols[f"{name}_nav"][i] = astro.navamsa_sign(lo)
            cols[f"{name}_lon"][i] = lo
        if logger and (i + 1) % 20000 == 0:
            spd = (i + 1) / (time.time() - t0)
            logger.info(f"[kundali-db] {i + 1}/{n} days "
                        f"(yr {cols['year'][i]}) | {spd:.0f} days/s")
    return cols


def write_duckdb(out_path: str, cols: dict, meta: dict, logger=None) -> str:
    """Write the columns to a DuckDB file and index every sign column."""
    import duckdb
    import pyarrow as pa

    table = pa.table({k: pa.array(v) for k, v in cols.items()})
    con = duckdb.connect(str(out_path))
    con.execute("DROP TABLE IF EXISTS positions")
    con.register("t_arrow", table)
    con.execute("CREATE TABLE positions AS SELECT * FROM t_arrow")
    con.unregister("t_arrow")
    for name in astro.BODY_NAMES:
        con.execute(f"CREATE INDEX idx_{name}_sign ON positions({name}_sign)")
    con.execute("CREATE TABLE meta (key VARCHAR, value VARCHAR)")
    con.executemany("INSERT INTO meta VALUES (?, ?)",
                    [(k, str(v)) for k, v in meta.items()])
    n = con.execute("SELECT COUNT(*) FROM positions").fetchone()[0]
    con.close()
    if logger:
        logger.info(f"[kundali-db] wrote {n} rows -> {out_path} "
                    f"(indexed {len(astro.BODY_NAMES)} sign columns)")
    return str(out_path)


def build_db(out_path: str, start_jd: float, end_jd: float, step_days: float = 1.0,
             logger=None) -> str:
    """End-to-end: compute daily sidereal positions and write the indexed DuckDB."""
    t0 = time.time()
    cols = build_arrays(start_jd, end_jd, step_days, logger=logger)
    meta = {
        "start_jd": start_jd, "end_jd": end_jd, "step_days": step_days,
        "n_days": cols["jd"].size, "ayanamsha": "lahiri",
        "bodies": ",".join(astro.BODY_NAMES),
        "start_year": jd_to_gregorian(start_jd)[0],
        "end_year": jd_to_gregorian(end_jd)[0],
        "build_seconds": round(time.time() - t0, 2),
    }
    return write_duckdb(out_path, cols, meta, logger)
