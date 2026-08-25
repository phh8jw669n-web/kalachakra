"""Kundali Twin Temporal Search Engine.

A standalone, high-speed temporal + geographic search over a daily sidereal
ephemeris (Swiss Ephemeris only — no neural network, no mesh). Finds the exact
historical days and longitudes where a user's astrological twin could have existed,
across eight escalating tiers of similarity.
"""

from .search import TIER_NAMES, KundaliSearch

__all__ = ["KundaliSearch", "TIER_NAMES"]
