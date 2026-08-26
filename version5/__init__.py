"""version5 — the GPU-native, client-side Kalachakra celestial weather engine.

This package completes the transition begun by ``train_v4``: the astrophysics is
fully **decoupled** from pixel rendering. A Transformer autoencoder is trained on a
Monte-Carlo random walk across the 10,256-year timeline, its 3-neuron OKLab
bottleneck is exported to **ONNX**, and the browser runs that neural field directly
on the GPU while a stateless FastAPI server only ever ships a ~2 KB payload of the
ten bodies' equatorial coordinates.

Design directives (from the version5 PRD):

* **Namespace isolation** — every new module lives here, under ``version5/``.
* **Zero code duplication** — the planetary math (``pyswisseph`` wrappers), the
  calendar, the Transformer encoder block and the OKLab colour maths are *imported*
  from the root ``kalachakra`` package, never copied.

Because ``version5`` is a root-level directory (a sibling of ``src/``), this
``__init__`` puts the ``src`` tree on ``sys.path`` so ``import kalachakra`` resolves
whether the project is pip-installed or run straight from a checkout — this also
covers ``uvicorn version5.server:app`` and ``python -m version5.train``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

__all__ = ["config", "ephemeris", "sky_math", "dataset", "model", "losses"]
