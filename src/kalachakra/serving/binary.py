"""
Binary field-frame packing for the WebSocket stream (blueprint §7).

Streaming high-dimensional field data as JSON would dominate the budget, so each
frame is packed into a compact little-endian ArrayBuffer the browser maps
directly into WebGL textures / uniform buffers with zero copies.

Layout (all little-endian, tightly packed):
    magic   uint32   0x4B434852  ('KCHR')
    version uint16   = 1
    flags   uint16   bit0 = latent block present
    n_nodes uint32
    latent  uint16   latent dim (0 if absent)
    ---- columns, each length n_nodes unless noted ----
    lat     float32
    lng     float32
    potential float32
    shear   float32
    macro   uint16
    micro   uint16
    [latent float32  (n_nodes * latent) if flag set]

Pure numpy; round-trip tested.
"""

from __future__ import annotations

import numpy as np

MAGIC = 0x4B434852
VERSION = 1
_HEADER = np.dtype([("magic", "<u4"), ("version", "<u2"), ("flags", "<u2"),
                    ("n_nodes", "<u4"), ("latent", "<u2")])


def pack_frame(lat: np.ndarray, lng: np.ndarray, potential: np.ndarray,
               shear: np.ndarray, macro: np.ndarray, micro: np.ndarray,
               latent: np.ndarray | None = None) -> bytes:
    """Pack one field frame into a binary buffer (see module layout)."""
    n = int(lat.shape[0])
    for name, a in (("lng", lng), ("potential", potential), ("shear", shear),
                    ("macro", macro), ("micro", micro)):
        if a.shape[0] != n:
            raise ValueError(f"{name} length {a.shape[0]} != n_nodes {n}")
    flags = 1 if latent is not None else 0
    latent_dim = int(latent.shape[1]) if latent is not None else 0

    header = np.zeros(1, dtype=_HEADER)
    header["magic"] = MAGIC
    header["version"] = VERSION
    header["flags"] = flags
    header["n_nodes"] = n
    header["latent"] = latent_dim

    parts = [
        header.tobytes(),
        np.asarray(lat, "<f4").tobytes(),
        np.asarray(lng, "<f4").tobytes(),
        np.asarray(potential, "<f4").tobytes(),
        np.asarray(shear, "<f4").tobytes(),
        np.asarray(macro, "<u2").tobytes(),
        np.asarray(micro, "<u2").tobytes(),
    ]
    if latent is not None:
        parts.append(np.ascontiguousarray(latent, "<f4").tobytes())
    return b"".join(parts)


def unpack_frame(buf: bytes) -> dict[str, np.ndarray]:
    """Inverse of :func:`pack_frame` (used by tests and Python clients)."""
    header = np.frombuffer(buf, dtype=_HEADER, count=1)[0]
    if int(header["magic"]) != MAGIC:
        raise ValueError("bad magic; not a Kalachakra field frame")
    n = int(header["n_nodes"])
    ld = int(header["latent"])
    off = _HEADER.itemsize

    def take(dtype, count):
        nonlocal off
        arr = np.frombuffer(buf, dtype=dtype, count=count, offset=off)
        off += arr.nbytes
        return arr.copy()

    out = {
        "lat": take("<f4", n), "lng": take("<f4", n),
        "potential": take("<f4", n), "shear": take("<f4", n),
        "macro": take("<u2", n), "micro": take("<u2", n),
    }
    if header["flags"] & 1:
        out["latent"] = take("<f4", n * ld).reshape(n, ld)
    return out
