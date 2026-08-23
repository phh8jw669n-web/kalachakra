#!/usr/bin/env python3
"""
Phase 1 — generate the memory-mapped global-state matrix (blueprint §1.3, §2, §3.2).

Computes G(t) frame-by-frame from DE441 via pyswisseph and serializes it to a
chunked, BF16, delta-encoded :class:`EphemerisStore`. The full run produces
~13.4 billion frames (~1.9 TB uncompressed BF16+delta); use ``--max-frames`` for
a bounded slice.

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
    p.add_argument("--start-date", type=str, default=None,
                   help="ISO datetime to start from (UTC); overrides --start-frame")
    p.add_argument("--start-frame", type=int, default=0,
                   help="frame index to start from (0 = Kali Yuga epoch, needs DE441)")
    p.add_argument("--chunk-frames", type=int, default=1_000_000,
                   help="frames per chunk file")
    p.add_argument("--max-frames", type=int, default=None,
                   help="number of frames to generate from the start")
    p.add_argument("--ephe-path", type=str, default=None,
                   help="directory holding the Swiss .se1 files, DE431 (enables full range; "
                        "without it the Moshier backend covers ~3000 BCE - 3000 CE)")
    p.add_argument("--jpl-file", type=str, default=None,
                   help="DE441 .bsp file for the JPL backend (full range)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if not global_state.ephemeris_available():
        print("ERROR: pyswisseph not installed. `pip install \"kalachakra[ephemeris]\"`",
              file=sys.stderr)
        return 2

    # Backend: explicit flag wins, else a saved config (setup_full_span), else Moshier.
    mode = global_state.configure_from_args(ephe_path=args.ephe_path,
                                            jpl_file=args.jpl_file)
    print(f"ephemeris backend: {mode}")

    # Resolve the start frame.
    if args.start_date:
        from kalachakra.ephemeris.calendar import parse_datetime
        start_frame = int(timeline.jd_to_frame(parse_datetime(args.start_date)))
    else:
        start_frame = args.start_frame

    timeline_total = C.total_temporal_frames()
    count = args.max_frames if args.max_frames is not None else \
        timeline_total - start_frame
    end_frame = min(start_frame + count, timeline_total)

    store = EphemerisStore(args.out)
    from kalachakra.ephemeris.calendar import format_jd
    print(f"Generating {end_frame - start_frame:,} frames -> {args.out}")
    print(f"  start frame {start_frame:,} = {format_jd(timeline.frame_to_jd(start_frame))}")
    print(f"  step {timeline.JD_STEP:.9f} d/frame ({C.VIGHATIKA_SECONDS}s)")

    written = 0
    t0 = time.monotonic()
    start = start_frame
    while start < end_frame:
        end = min(start + args.chunk_frames, end_frame)
        idx = np.arange(start, end)
        jds = timeline.frame_to_jd(idx)
        frames = global_state.global_state_batch(jds)         # (n, 10, 7)
        store.write_chunk(start, frames.astype(np.float32))
        written += end - start
        rate = written / max(time.monotonic() - t0, 1e-9)
        print(f"  chunk [{start:,}:{end:,}] written "
              f"({written:,}/{end_frame - start_frame:,}, {rate:,.0f} frames/s)")
        start = end

    print(f"Done. {written:,} frames in {len(store.chunks())} chunks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
