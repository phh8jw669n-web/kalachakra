"""The micro-payload backend.

A stateless FastAPI server whose only job is astronomy: for a requested timestamp it
runs the single ten-call ephemeris query and returns a **< 2 KB** JSON payload of the
ten bodies' equatorial coordinates plus the sidereal time. No PyTorch, no grids, no
pixels — all neural inference and rendering happen on the client GPU. It also serves
the static frontend (``index.html``, the ``.onnx`` model, ``golden.json``).

Run it:
    uvicorn version5.server:app --reload
    # or, to pin the deep-time backend and choose host/port:
    python -m version5.server --ephe-path /path/to/ephe --port 8000

The ephemeris backend is configured once at import from the usual environment /
config (``KALACHAKRA_EPHE_PATH`` etc. — see kalachakra.ephemeris.global_state), so
importing under uvicorn "just works".
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi import FastAPI, HTTPException, Query          # noqa: E402
from fastapi.middleware.cors import CORSMiddleware          # noqa: E402
from fastapi.responses import JSONResponse                  # noqa: E402
from fastapi.staticfiles import StaticFiles                 # noqa: E402

from kalachakra.ephemeris import global_state               # noqa: E402
from kalachakra.ephemeris.calendar import format_jd, parse_datetime  # noqa: E402

from version5 import ephemeris as ephem                     # noqa: E402

WEB_DIR = Path(__file__).resolve().parent / "web"

app = FastAPI(title="Kalachakra version5 — telemetry", version="5.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

# Configure the ephemeris backend exactly once, at import (covers uvicorn).
_BACKEND = "unavailable"
if global_state.ephemeris_available():
    _BACKEND = ephem.configure(ephe_path=os.environ.get("KALACHAKRA_EPHE_PATH"),
                               jpl_file=os.environ.get("KALACHAKRA_JPL_FILE"))


@app.get("/telemetry")
def telemetry(time: str | None = Query(None, description="UTC ISO-8601; default now")):
    """The micro-payload: RA/Dec of the ten bodies + GAST for one instant."""
    if _BACKEND == "unavailable":
        raise HTTPException(503, "pyswisseph is not installed on the server.")
    try:
        jd = parse_datetime(time) if time else parse_datetime("now")
    except ValueError as exc:
        raise HTTPException(400, f"bad time: {exc}") from exc
    payload = ephem.telemetry(jd)
    payload["timestamp"] = format_jd(jd)
    payload["backend"] = _BACKEND
    return JSONResponse(payload)


@app.get("/api/info")
def info():
    """Server capabilities (handy for the frontend to show which backend is live)."""
    return {
        "service": "kalachakra-version5",
        "backend": _BACKEND,
        "bodies": list(ephem.BODY_NAMES),
        "has_model": (WEB_DIR / "model_v5.onnx").is_file(),
        "has_golden": (WEB_DIR / "golden.json").is_file(),
    }


# Static frontend LAST so the explicit API routes above take precedence over the
# catch-all mount at "/". html=True serves index.html at the root.
if WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Run the version5 telemetry server.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--ephe-path", default=None, help="Swiss .se1 dir (full span)")
    p.add_argument("--jpl-file", default=None, help="JPL DE441 .bsp file")
    p.add_argument("--reload", action="store_true")
    args = p.parse_args(argv)
    # Push CLI backend choice into the env so the reconfigured import picks it up.
    if args.ephe_path:
        os.environ["KALACHAKRA_EPHE_PATH"] = args.ephe_path
    if args.jpl_file:
        os.environ["KALACHAKRA_JPL_FILE"] = args.jpl_file
    global _BACKEND
    if global_state.ephemeris_available():
        _BACKEND = ephem.configure(ephe_path=args.ephe_path, jpl_file=args.jpl_file)
    import uvicorn
    uvicorn.run("version5.server:app" if args.reload else app,
                host=args.host, port=args.port, reload=args.reload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
