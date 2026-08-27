"""Kalachakra version6 — the continuous SIREN engine.

A ground-up rebuild that abandons every grid, texture and time-step of version5. The
whole system is *continuous*:

* a self-contained analytic **topocentric ephemeris** turns any ``(lat, lon, jd)`` into
  a flat 33-D local-sky tensor (11 bodies x North/East/Up unit vectors),
* a **SIREN** (sinusoidal representation network) maps that 33-D physical state to a
  3-D CIE L*a*b* colour, trained by an **isometric** loss (physical distance == colour
  distance),
* the trained SIREN weights are exported and the *entire* ephemeris + network is
  re-run per-pixel inside a GLSL fragment shader, so the globe has infinite resolution.

The ephemeris is deliberately dependency-free (closed-form Kepler elements, no
``pyswisseph``) so the *identical* maths runs in Python (training), JavaScript (HUD)
and GLSL (rendering). This trades Swiss-Ephemeris arc-second accuracy for perfect
cross-platform continuity — which is what a "geometric mirror" needs.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

__all__ = ["ephemeris", "siren", "dataset", "losses"]
