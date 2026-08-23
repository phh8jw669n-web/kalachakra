"""
Swiss Ephemeris ``.se1`` file planning.

Each Swiss Ephemeris file covers exactly 600 years. The filename suffix encodes
the 600-year block: ``_NN`` is a CE block, ``mNN`` a BCE block, and the number
times 100 is the start century (``_00`` = 1 BCE-599 CE, ``_18`` = 1800-2399 CE,
``m36`` = 3601-3002 BCE). Files are grouped by body:

    sepl* = planets (Mercury..Pluto)      semo* = Moon
    seas* = main asteroids (NOT needed by Kalachakra)

Kalachakra uses the Sun, Moon, planets and the lunar nodes (Rahu/Ketu), which
require only ``sepl*`` and ``semo*``.

This module computes exactly which blocks/filenames cover a given year span, so
the setup script downloads the minimal correct set. Years are astronomical
(1 BCE == 0, 2 BCE == -1, ...). Pure stdlib; fully unit-tested.
"""

from __future__ import annotations

from dataclasses import dataclass

BLOCK_YEARS = 600

# Body-group prefixes we need (asteroids deliberately excluded).
DEFAULT_PREFIXES = ("sepl", "semo")

# Swiss Ephemeris (DE431) validity, astronomical years (~13000 BCE .. 16800 CE).
SE_MIN_YEAR = -12999
SE_MAX_YEAR = 16800

# Kalachakra timeline in astronomical years: 3102 BCE .. 7154 CE.
KALACHAKRA_START_YEAR = -3101
KALACHAKRA_END_YEAR = 7154


@dataclass(frozen=True)
class Block:
    tag: str            # e.g. "_18" or "m36"
    start_year: int     # astronomical year (inclusive)
    end_year: int       # astronomical year (inclusive)

    def filename(self, prefix: str) -> str:
        return f"{prefix}{self.tag}.se1"


def _tag_for_start(start_astro: int) -> str:
    """Filename tag for a block whose first astronomical year is ``start_astro``."""
    if start_astro < 0:
        return f"m{(-start_astro) // 100:02d}"
    return f"_{start_astro // 100:02d}"


def all_blocks() -> list[Block]:
    """Every 600-year block across the Swiss Ephemeris validity range."""
    blocks = []
    start = -13200  # aligned 600-year grid boundary just below SE_MIN_YEAR
    while start <= SE_MAX_YEAR:
        blocks.append(Block(_tag_for_start(start), start, start + BLOCK_YEARS - 1))
        start += BLOCK_YEARS
    return blocks


def blocks_for_years(start_year: int = KALACHAKRA_START_YEAR,
                     end_year: int = KALACHAKRA_END_YEAR) -> list[Block]:
    """Blocks (in time order) that overlap the inclusive span [start, end]."""
    if start_year > end_year:
        raise ValueError("start_year must be <= end_year")
    return [b for b in all_blocks()
            if not (b.end_year < start_year or b.start_year > end_year)]


def filenames_for_years(start_year: int = KALACHAKRA_START_YEAR,
                        end_year: int = KALACHAKRA_END_YEAR,
                        prefixes: tuple[str, ...] = DEFAULT_PREFIXES) -> list[str]:
    """Minimal ``.se1`` filenames covering the span, for the given body groups."""
    return [b.filename(p) for b in blocks_for_years(start_year, end_year)
            for p in prefixes]


def fmt_year(astro: int) -> str:
    """Human label for an astronomical year (0 == 1 BCE)."""
    return f"{astro} CE" if astro >= 1 else f"{1 - astro} BCE"
