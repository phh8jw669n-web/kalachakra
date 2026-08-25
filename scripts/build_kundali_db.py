#!/usr/bin/env python3
"""
Build the daily sidereal-ephemeris DuckDB for the Kundali Twin search engine.

One row per day holds the nine grahas' Lahiri-sidereal sign / degree / nakshatra /
navamsa across the requested span, with every sign column indexed so the eight
tier sweeps resolve in milliseconds. Independent of the neural core.

The full Holocene needs a wide-range backend (DE441 / Swiss files installed); the
Moshier default only covers ~3000 BCE-3000 CE, so pass a span your ephemeris
covers. Astronomical years: 0 == 1 BCE, -9999 == 10000 BCE.

Example (full ~10k-year span, needs DE441):
    python scripts/build_kundali_db.py --start-year -8000 --end-year 2100 --out kundali.duckdb
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="kundali.duckdb")
    p.add_argument("--start-year", type=int, default=1800, help="astronomical year (0==1 BCE)")
    p.add_argument("--end-year", type=int, default=2100)
    p.add_argument("--step-days", type=float, default=1.0)
    args = p.parse_args(argv)

    from kalachakra.ephemeris import global_state
    from kalachakra.ephemeris.calendar import gregorian_to_jd
    from kalachakra.kundali import db

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                        datefmt="%H:%M:%S")
    logger = logging.getLogger("kundali-build")

    if not global_state.ephemeris_available():
        print("ERROR: pyswisseph unavailable.", file=sys.stderr)
        return 2
    mode = global_state.auto_configure()
    logger.info(f"ephemeris backend: {mode}")
    if mode == "moshier" and (args.start_year < -3000 or args.end_year > 3000):
        logger.warning("Moshier backend only covers ~3000 BCE-3000 CE; positions "
                       "outside that range will be unreliable. Install DE441/Swiss "
                       "files for the full Holocene.")

    s = gregorian_to_jd(args.start_year, 1, 1)
    e = gregorian_to_jd(args.end_year, 1, 1)
    logger.info(f"building {args.start_year}..{args.end_year} "
                f"(~{int((e - s) / args.step_days):,} rows) -> {args.out}")
    t0 = time.time()
    db.build_db(args.out, s, e, step_days=args.step_days, logger=logger)
    logger.info(f"done in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
