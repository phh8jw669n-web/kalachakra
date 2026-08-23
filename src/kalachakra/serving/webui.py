"""
Shared web-UI plumbing for the serving apps: permissive CORS (so a browser page
served from any origin can reach the API) and a same-origin static mount of the
repo's ``web/`` directory (so the visuals load with no CORS at all).

Both helpers are no-ops when FastAPI is missing or the directory is absent, so
importing/serving never hard-fails on an optional dependency.
"""

from __future__ import annotations

from pathlib import Path


def enable_cors(app) -> None:
    """Allow cross-origin browser fetches (localhost dev + file:// pages)."""
    try:  # pragma: no cover - trivial passthrough
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"], allow_credentials=False,
            allow_methods=["*"], allow_headers=["*"],
        )
    except Exception:  # noqa: BLE001
        pass


def default_web_dir() -> Path:
    """The repo's ``web/`` directory (editable installs / source checkouts)."""
    # src/kalachakra/serving/webui.py -> parents[3] == repo root
    return Path(__file__).resolve().parents[3] / "web"


def mount_web_ui(app, web_dir: str | Path | None = None, path: str = "/ui") -> Path | None:
    """Mount ``web_dir`` (default: the repo ``web/``) as static files at ``path``.

    Returns the mounted directory, or ``None`` if it does not exist. Serving the
    HTML from the same origin as the API means the visuals need no CORS: open
    ``http://HOST:PORT/ui/index.html`` (globe) or ``/ui/radar.html`` (radar).
    """
    web_dir = Path(web_dir) if web_dir is not None else default_web_dir()
    if not web_dir.is_dir():
        return None
    try:
        from fastapi.staticfiles import StaticFiles

        app.mount(path, StaticFiles(directory=str(web_dir), html=True), name="ui")
        return web_dir
    except Exception:  # noqa: BLE001
        return None
