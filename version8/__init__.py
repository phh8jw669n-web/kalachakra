"""Kalachakra version8 — the 88-D relational SIREN engine.

A self-contained rebuild (no cross-version imports) that adds explicit *relational awareness*
to the continuous colour field:

* an **88-D state** = 33-D topocentric local vectors (11 bodies x North/East/Zenith) plus the
  55-D **geometric chords** (all pairwise dot products, i.e. mutual angular separations),
* a **4x128 SIREN** with a gamut-bounded head (L* in (5,95), a*/b* in (-80,80)) so colour can
  never clamp to pure white or black,
* a **balanced isometric loss** that RMS-normalises the local (33) and chord (55) distances so
  neither dominates, and
* a **vertex-shader** renderer that runs the whole 88-D network per vertex (singularity-free
  local basis from the surface normal) and interpolates colour across triangles for 60+ fps.

Everything (Python ephemeris + the web ES modules) is copied into this folder so version8 can
be trained and served on its own, with no dependency on other version folders.
"""

from __future__ import annotations

__all__ = ["ephemeris", "state", "siren", "losses", "config", "dataset", "training"]
