#!/usr/bin/env python3
"""
Phase 1 — generate the memory-mapped global-state matrix (blueprint §1.3, §2, §3.2).

Computes G(t) frame-by-frame from DE441 via pyswisseph and serializes it to a
chunked, BF16, delta-encoded :class:`EphemerisStore`. The full run produces
~13.4 billion frames (~300 GB); use ``--max-frames`` for a bounded slice.

Requires:  pip install "kalachakra[ephemeris]"   (pyswisseph + DE441 data files)

Example (first 10,000 frames, 5,000 per chunk):
    python scripts/generate_ephemeris.py --out data/ephemeris \\
        --max-frames 10000 --chunk-frames 5000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kalachakra import constants as C                       # noqa: E402
from kalachakra.ephemeris import global_state, timeline     # noqa: E402
from kalachakra.storage.binary_store import EphemerisStore  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("data/ephemeris"),
                   help="output store directory")
    p.add_argument("--chunk-frames", type=int, default=1_000_000,
                   help="frames per chunk file")
    p.add_argument("--max-frames", type=int, default=None,
                   help="stop after this many frames (default: full timeline)")
    p.add_argument("--ephe-path", type=str, default=None,
                   help="directory holding the DE441 ephemeris data files")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if not global_state.ephemeris_available():
        print("ERROR: pyswisseph not installed. `pip install \"kalachakra[ephemeris]\"`",
              file=sys.stderr)
        return 2

    import swisseph as swe
    if args.ephe_path:
        swe.set_ephe_path(args.ephe_path)

    total = C.total_temporal_frames()
    if args.max_frames is not None:
        total = min(total, args.max_frames)

    store = EphemerisStore(args.out)
    print(f"Generating {total:,} frames -> {args.out}")
    print(f"  epoch JD {C.KALI_YUGA_EPOCH_JD} + {timeline.JD_STEP:.9f} d/frame")

    written = 0
    t0 = time.monotonic()
    for start, end in timeline.iter_chunk_ranges(args.chunk_frames):
        if start >= total:
            break
        end = min(end, total)
        idx = np.arange(start, end)
        jds = timeline.frame_to_jd(idx)
        frames = global_state.global_state_batch(jds)         # (n, 10, 7)
        store.write_chunk(start, frames.astype(np.float32))
        written += end - start
        rate = written / max(time.monotonic() - t0, 1e-9)
        print(f"  chunk [{start:,}:{end:,}] written "
              f"({written:,}/{total:,}, {rate:,.0f} frames/s)")

    print(f"Done. {written:,} frames in {len(store.chunks())} chunks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
