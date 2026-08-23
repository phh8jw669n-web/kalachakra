"""Isomorphic Mathematical Transducer: physics-based, losslessly invertible rendering."""
from . import photometric, spectral, kinematics
__all__ = ["photometric", "spectral", "kinematics", "topography", "state"]

def __getattr__(name):  # topography/state require scipy — import lazily
    if name in ("topography", "state"):
        import importlib
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(name)
