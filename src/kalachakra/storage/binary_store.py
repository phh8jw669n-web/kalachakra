"""
Memory-mapped binary storage for the global-state timeline (blueprint §3.2).

The 13.4-billion-frame G(t) matrix is serialized to contiguous ``.mmap`` chunk
files in BF16 half precision with temporal delta encoding. This module provides:

* lossy ``float32 <-> bfloat16`` conversion via bit truncation with
  round-to-nearest-even (numpy has no native bf16 dtype);
* reversible temporal delta encoding (store a base frame, then frame-to-frame
  differences) which shrinks the smooth planetary trajectories dramatically;
* :class:`EphemerisStore`, a chunked, memory-mapped reader/writer.

Everything here is pure numpy and round-trip tested.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from .. import constants as C


# ---------------------------------------------------------------------------
# BF16 <-> float32 (bit-level; no native numpy bf16 dtype)
# ---------------------------------------------------------------------------

def float32_to_bf16(x: np.ndarray) -> np.ndarray:
    """Truncate float32 to bfloat16 (top 16 bits), round-to-nearest-even.

    Returns a ``uint16`` array holding the bf16 bit patterns.
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    u = x.view(np.uint32)
    # Round-to-nearest-even: add 0x7FFF plus the lsb of the retained mantissa.
    lsb = (u >> np.uint32(16)) & np.uint32(1)
    bias = np.uint32(0x7FFF) + lsb
    rounded = u + bias
    return (rounded >> np.uint32(16)).astype(np.uint16)


def bf16_to_float32(b: np.ndarray) -> np.ndarray:
    """Expand bfloat16 bit patterns (uint16) back to float32."""
    b = np.ascontiguousarray(b, dtype=np.uint16)
    u = b.astype(np.uint32) << np.uint32(16)
    return u.view(np.float32)


# ---------------------------------------------------------------------------
# Temporal delta encoding
# ---------------------------------------------------------------------------

def delta_encode(frames: np.ndarray) -> np.ndarray:
    """Replace ``frames[k]`` (k>0) with ``frames[k] - frames[k-1]``.

    Frame 0 is kept as the absolute base. Operates along axis 0.
    """
    frames = np.asarray(frames, dtype=np.float32)
    out = np.empty_like(frames)
    out[0] = frames[0]
    if frames.shape[0] > 1:
        out[1:] = np.diff(frames, axis=0)
    return out


def delta_decode(deltas: np.ndarray) -> np.ndarray:
    """Inverse of :func:`delta_encode` (cumulative sum along axis 0)."""
    return np.cumsum(np.asarray(deltas, dtype=np.float32), axis=0)


# ---------------------------------------------------------------------------
# Chunked, memory-mapped store
# ---------------------------------------------------------------------------

@dataclass
class ChunkMeta:
    """Sidecar metadata for one ``.mmap`` chunk."""

    start_frame: int
    n_frames: int
    shape_tail: tuple[int, ...]   # per-frame shape, e.g. (10, 7)
    dtype: str                    # always "bfloat16-uint16"
    delta_encoded: bool


class EphemerisStore:
    """Reader/writer for the delta-encoded, BF16, memory-mapped ephemeris.

    Layout on disk::

        root/
          manifest.json          # list of ChunkMeta
          chunk_00000000.mmap    # uint16 bf16 payload, delta-encoded
          chunk_00000000.json    # ChunkMeta sidecar
          ...

    Frames are addressed by absolute frame index across the whole timeline.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # -- writing ----------------------------------------------------------
    def _chunk_stub(self, start_frame: int) -> Path:
        return self.root / f"chunk_{start_frame:012d}"

    def write_chunk(self, start_frame: int, frames: np.ndarray,
                    *, delta_encoded: bool = True) -> ChunkMeta:
        """Serialize ``frames`` (shape ``(n, *tail)``) as one chunk file."""
        frames = np.asarray(frames, dtype=np.float32)
        payload = delta_encode(frames) if delta_encoded else frames
        bits = float32_to_bf16(payload).reshape(frames.shape[0], -1)

        stub = self._chunk_stub(start_frame)
        mm = np.memmap(stub.with_suffix(C.BINARY_CHUNK_EXTENSION),
                       dtype=np.uint16, mode="w+", shape=bits.shape)
        mm[:] = bits
        mm.flush()
        del mm

        meta = ChunkMeta(
            start_frame=start_frame,
            n_frames=int(frames.shape[0]),
            shape_tail=tuple(int(s) for s in frames.shape[1:]),
            dtype="bfloat16-uint16",
            delta_encoded=delta_encoded,
        )
        stub.with_suffix(".json").write_text(json.dumps(asdict(meta)))
        self._append_manifest(meta)
        return meta

    def _append_manifest(self, meta: ChunkMeta) -> None:
        manifest = self._load_manifest()
        manifest = [m for m in manifest if m["start_frame"] != meta.start_frame]
        manifest.append(asdict(meta))
        manifest.sort(key=lambda m: m["start_frame"])
        (self.root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    def _load_manifest(self) -> list[dict]:
        path = self.root / "manifest.json"
        if not path.exists():
            return []
        return json.loads(path.read_text())

    # -- reading ----------------------------------------------------------
    def read_chunk(self, start_frame: int) -> np.ndarray:
        """Load and decode one chunk back to float32 ``(n, *tail)``."""
        stub = self._chunk_stub(start_frame)
        meta = ChunkMeta(**json.loads(stub.with_suffix(".json").read_text()))
        tail = tuple(meta.shape_tail)
        flat = int(np.prod(tail)) if tail else 1
        mm = np.memmap(stub.with_suffix(C.BINARY_CHUNK_EXTENSION),
                       dtype=np.uint16, mode="r",
                       shape=(meta.n_frames, flat))
        frames = bf16_to_float32(np.asarray(mm)).reshape((meta.n_frames, *tail))
        if meta.delta_encoded:
            frames = delta_decode(frames)
        return frames

    def chunks(self) -> list[ChunkMeta]:
        return [ChunkMeta(**m) for m in self._load_manifest()]
