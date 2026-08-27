"""Kalachakra version7 — the macro-scale regional & city-grid engine.

Version 6 proved the continuous SIREN colour field but paid for it at render time: the
*entire* ephemeris + network ran per pixel in a fragment shader, so a large window at high
DPI evaluated millions of transcendental sines every frame and the UI lagged.

Version 7 keeps the exact same physics and the exact same learned colour field but changes
*where* it is evaluated. Instead of an infinite per-pixel field it samples a structured,
high-density global grid — a regional lat/lon lattice plus a curated set of major
metropolitan hubs — bakes those node colours into a small equirectangular texture on a
background worker, and the globe simply *maps that texture* onto a sphere. Rendering is then
a trivial texture lookup that holds 60 fps at any zoom or resolution, while the field texture
refreshes off the main thread as time advances.

Reuse (no redundancy): the topocentric ephemeris, the SIREN with its bounded/soft-clamped
L*a*b* head, the isometric loss, and the calendar/colour utilities are imported directly
from :mod:`version6` — Python for training, and the very same ``version6/web`` ES modules
for the browser field worker and HUD. Version 7 adds only the structured node dataset, the
grid/city manifest, and the texture-mapping frontend.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

__all__ = ["cities", "config", "dataset", "training"]
