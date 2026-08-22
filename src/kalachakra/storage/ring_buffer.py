"""
Asynchronous ring buffer for ephemeris chunk prefetch (blueprint §3.2).

A bounded, thread-backed prefetcher that reads upcoming chunks from an
:class:`~kalachakra.storage.binary_store.EphemerisStore` ahead of the training
loop so the GPU never stalls on disk I/O. In production this partition is capped
at ``RING_BUFFER_BUDGET_GB`` (20 GB) of unified memory; here ``max_prefetch``
bounds the queue depth in *chunks*.

The buffer evicts consumed chunks automatically (the queue simply moves on) and
streams the next epoch's chunks in the background — the eviction/anticipation
behavior the blueprint requires, in a small, dependency-free form.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterator, Sequence

import numpy as np

from .binary_store import EphemerisStore


class RingBuffer:
    """Background prefetcher yielding decoded chunks in timeline order."""

    def __init__(self, store: EphemerisStore, start_frames: Sequence[int],
                 *, max_prefetch: int = 3):
        self._store = store
        self._order = list(start_frames)
        self._q: queue.Queue = queue.Queue(maxsize=max(1, max_prefetch))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _worker(self) -> None:
        for start_frame in self._order:
            if self._stop.is_set():
                break
            try:
                chunk = self._store.read_chunk(start_frame)
            except Exception as exc:  # noqa: BLE001 - surface to consumer
                self._q.put(("error", exc))
                return
            # Blocks when the buffer is full -> natural back-pressure / eviction.
            while not self._stop.is_set():
                try:
                    self._q.put(("chunk", (start_frame, chunk)), timeout=0.1)
                    break
                except queue.Full:
                    continue
        self._q.put(("done", None))

    def __enter__(self) -> "RingBuffer":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("RingBuffer already started")
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        while True:
            tag, payload = self._q.get()
            if tag == "done":
                return
            if tag == "error":
                raise payload  # re-raise worker exception on the consumer thread
            yield payload

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
