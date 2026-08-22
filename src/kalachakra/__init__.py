"""
Project Kalachakra — an autonomous, unsupervised ML system that maps the
continuous spatio-temporal geometric waves of the solar system onto a geodesic
model of Earth (see ARCHITECTURE.md).

The top-level package intentionally imports only dependency-light modules
(``constants``, ``geometry``) so ``import kalachakra`` works everywhere. Heavy
subsystems (``kalachakra.models``, ``kalachakra.training``, ``kalachakra.data``)
require PyTorch and are imported explicitly by callers that need them.
"""

from __future__ import annotations

from . import constants, geometry

__version__ = "0.1.0"

__all__ = ["constants", "geometry", "__version__"]
