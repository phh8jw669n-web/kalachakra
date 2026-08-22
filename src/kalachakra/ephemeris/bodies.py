"""
The ten active celestial entities of the global state vector G(t) (blueprint §2.3).

Seven observable masses, the two lunar-node vectors (Rahu / Ketu), and the
precession (Ayanamsha) vector. Swiss Ephemeris body identifiers are recorded here
so :mod:`kalachakra.ephemeris.global_state` can query DE441 without hard-coding
integers at the call site.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Kind(IntEnum):
    """How a body's ecliptic coordinates are obtained."""

    BODY = 0        #: direct pyswisseph body query
    NODE = 1        #: derived lunar node (true node / node + 180 deg)
    PRECESSION = 2  #: the Ayanamsha angle, not a physical body


@dataclass(frozen=True)
class CelestialEntity:
    """One row of the G(t) matrix."""

    name: str
    kind: Kind
    #: Swiss Ephemeris body id (``swe.SUN`` etc.), or ``None`` for derived rows.
    swe_id: int | None = None
    #: Fixed offset in degrees added to the parent longitude (Ketu = Rahu + 180).
    longitude_offset_deg: float = 0.0


# Swiss Ephemeris integer ids (mirrored so `import swisseph` is not required just
# to describe the body table). Values match pyswisseph's public constants.
_SWE_SUN = 0
_SWE_MOON = 1
_SWE_MERCURY = 2
_SWE_VENUS = 3
_SWE_MARS = 4
_SWE_JUPITER = 5
_SWE_SATURN = 6
_SWE_TRUE_NODE = 11


#: Canonical, order-stable table. Index == row index in G(t) (0..9).
ENTITIES: tuple[CelestialEntity, ...] = (
    CelestialEntity("Sun", Kind.BODY, _SWE_SUN),
    CelestialEntity("Moon", Kind.BODY, _SWE_MOON),
    CelestialEntity("Mercury", Kind.BODY, _SWE_MERCURY),
    CelestialEntity("Venus", Kind.BODY, _SWE_VENUS),
    CelestialEntity("Mars", Kind.BODY, _SWE_MARS),
    CelestialEntity("Jupiter", Kind.BODY, _SWE_JUPITER),
    CelestialEntity("Saturn", Kind.BODY, _SWE_SATURN),
    CelestialEntity("Rahu", Kind.NODE, _SWE_TRUE_NODE, longitude_offset_deg=0.0),
    CelestialEntity("Ketu", Kind.NODE, _SWE_TRUE_NODE, longitude_offset_deg=180.0),
    CelestialEntity("Ayanamsha", Kind.PRECESSION, None),
)

NAMES: tuple[str, ...] = tuple(e.name for e in ENTITIES)


def index_of(name: str) -> int:
    """Row index of the named entity in G(t)."""
    return NAMES.index(name)
