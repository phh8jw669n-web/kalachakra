"""State-lock checkpointing and atomic writes for crash-safe indexing (PRD page 6).

Every durable write goes to a temp path in the same directory and is ``os.replace``-d
into place only once fully written, so a crash mid-write never corrupts the primary
dataset (the fragmented temp file is simply discarded on recovery). The run's
progress lives in a single JSON state file that is itself written atomically; on
reboot the recovery protocol reads it and resumes at the next unprocessed chunk,
skipping all completed work.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically (temp file in the same dir + replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=path.suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)                 # atomic on POSIX and Windows
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def atomic_write_text(path: str | Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


class StateLock:
    """The recovery ledger: which phases/chunks are done, and where to resume."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.data: dict = self._load()

    def _load(self) -> dict:
        try:
            return json.loads(self.path.read_text())
        except Exception:  # noqa: BLE001
            return {"phases": {}, "chunks_done": [], "config": {}}

    def flush(self) -> None:
        atomic_write_text(self.path, json.dumps(self.data, indent=2, sort_keys=True))

    # -- config provenance ---------------------------------------------------
    def set_config(self, cfg: dict) -> None:
        self.data["config"] = cfg
        self.flush()

    # -- phase completion ----------------------------------------------------
    def phase_done(self, name: str) -> bool:
        return bool(self.data.get("phases", {}).get(name, {}).get("done"))

    def mark_phase(self, name: str, **info) -> None:
        self.data.setdefault("phases", {})[name] = {"done": True, **info}
        self.flush()

    # -- per-chunk completion (Phase 2 temporal sweep) -----------------------
    def chunk_done(self, chunk_id: int) -> bool:
        return chunk_id in set(self.data.get("chunks_done", []))

    def mark_chunk(self, chunk_id: int, **info) -> None:
        done = set(self.data.get("chunks_done", []))
        done.add(int(chunk_id))
        self.data["chunks_done"] = sorted(done)
        self.data.setdefault("chunk_info", {})[str(chunk_id)] = info
        self.flush()

    def last_chunk(self) -> int:
        done = self.data.get("chunks_done", [])
        return max(done) if done else -1
