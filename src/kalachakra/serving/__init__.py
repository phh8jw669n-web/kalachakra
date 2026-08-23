"""Phase 4 serving: broadcast engine, binary framing, FastAPI app (blueprint §7)."""
from . import broadcast, binary
__all__ = ["broadcast", "binary", "app"]

def __getattr__(name):  # lazy: app imports fastapi (optional)
    if name == "app":
        from . import app as _app
        return _app
    raise AttributeError(name)
