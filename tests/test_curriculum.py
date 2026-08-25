"""Curriculum learning: the progressive multi-scale temporal-resolution schedule.

Covers the per-epoch stride schedule, the continuous vs. random-micro-burst window
generators (pure JD math, no ephemeris), and one end-to-end projection check that
the yielded windows match the shape the v3 trainer consumes.
"""

import itertools

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from kalachakra import constants as C                                # noqa: E402
from kalachakra.data.curriculum import (                            # noqa: E402
    SUB_HOUR_THRESHOLD_S, CurriculumPhase, CurriculumStream, curriculum_phase,
)
from kalachakra.data.dataset import StreamConfig                    # noqa: E402
from kalachakra.ephemeris import global_state                       # noqa: E402
from kalachakra.ephemeris.calendar import parse_datetime           # noqa: E402
from kalachakra.grid import geodesic                                # noqa: E402

_HOUR = 3600.0
_MIN = 60.0


def test_curriculum_schedule():
    """Each phase has the stride / mode / burst plan the PRD specifies."""
    solar = curriculum_phase(0)
    assert solar == curriculum_phase(4)             # 0..4 all Solar
    assert solar.name == "solar" and solar.stride_seconds == 24 * _HOUR
    assert solar.mode == "continuous"

    asc = curriculum_phase(5)
    assert asc == curriculum_phase(9) and asc.name == "ascendant"
    assert asc.stride_seconds == 2 * _HOUR and asc.mode == "continuous"

    nav = curriculum_phase(10)
    assert nav == curriculum_phase(14) and nav.name == "navamsha"
    assert nav.stride_seconds == 24 * _MIN and nav.mode == "microburst"
    assert nav.n_windows == 1000 and 180 < nav.window_span_days < 185   # ~6 months

    deg = curriculum_phase(15)
    assert deg == curriculum_phase(19) and deg.name == "degree"
    assert deg.stride_seconds == 4 * _MIN and deg.mode == "microburst"
    assert deg.n_windows == 2000 and 29 < deg.window_span_days < 32     # ~1 month

    quantum = curriculum_phase(20)
    assert quantum == curriculum_phase(999)         # 20+ all Quantum
    assert quantum.stride_seconds == float(C.VIGHATIKA_SECONDS) == 24.0
    assert quantum.mode == "microburst"
    assert quantum.n_windows == 5000 and quantum.window_span_days == 7.0


def test_micro_bursting_constraint():
    """The crucial constraint: stride < 1 h => micro-burst, else continuous sweep."""
    for epoch in range(0, 40):
        ph = curriculum_phase(epoch)
        if ph.stride_seconds < SUB_HOUR_THRESHOLD_S:
            assert ph.mode == "microburst" and ph.n_windows > 0
            assert ph.window_span_days > 0
        else:
            assert ph.mode == "continuous"


def _stream(**kw):
    grid = geodesic.fibonacci_sphere(48)
    cfg = StreamConfig(window_frames=kw.pop("T", 4), window_stride=kw.pop("hop", 4))
    return CurriculumStream(grid, cfg, **kw)


def test_continuous_window_math():
    """Continuous sweep: windows tile the range at exactly the target stride."""
    s = _stream(start_jd=1000.0, end_jd=1010.0, T=3, hop=3)   # 24 h stride @ epoch 0
    s.set_epoch(0)
    wins = list(s._continuous_windows(curriculum_phase(0), wid=0, nw=1))
    assert wins
    stride_days = 24 * _HOUR / C.SECONDS_PER_DAY              # == 1.0 day
    for jds in wins:
        assert len(jds) == 3
        assert np.allclose(np.diff(jds), stride_days)
    assert wins[0][0] == 1000.0                               # starts at range start
    # worker sharding partitions the windows with no overlap and no loss
    a = [w[0] for w in s._continuous_windows(curriculum_phase(0), 0, 2)]
    b = [w[0] for w in s._continuous_windows(curriculum_phase(0), 1, 2)]
    allw = [w[0] for w in wins]
    assert sorted(a + b) == allw and not (set(a) & set(b))


def test_microburst_window_math():
    """Micro-burst: bounded random windows at the fine stride, reproducible."""
    phase = CurriculumPhase("t", 24.0, "microburst", n_windows=6,
                            window_span_days=1.0)             # 24 s stride, 1-day span
    s = _stream(start_jd=2_400_000.0, end_jd=2_400_050.0, T=4, hop=50)
    s.set_epoch(20)
    wins = list(s._microburst_windows(phase, wid=0, nw=1))
    assert wins
    stride_days = 24.0 / C.SECONDS_PER_DAY
    for jds in wins:
        assert len(jds) == 4
        assert np.allclose(np.diff(jds), stride_days)
        # every window lies inside a [start, end-span] anchored 1-day burst
        assert 2_400_000.0 <= jds[0] <= 2_400_050.0
    # deterministic for a fixed (seed, epoch)
    again = list(s._microburst_windows(phase, wid=0, nw=1))
    assert np.array_equal(np.concatenate(wins), np.concatenate(again))
    # a different epoch reseeds the anchors -> different windows
    s.set_epoch(21)
    other = list(s._microburst_windows(phase, wid=0, nw=1))
    assert not np.array_equal(np.concatenate(wins), np.concatenate(other))
    # bursts are sharded across workers (union == all, disjoint)
    s.set_epoch(20)
    w0 = list(s._microburst_windows(phase, 0, 2))
    w1 = list(s._microburst_windows(phase, 1, 2))
    assert len(w0) + len(w1) == len(wins)


def test_curriculum_emit_shapes():
    """End to end: yielded windows match the (T, N, B*5) tensors the trainer eats."""
    if not global_state.ephemeris_available():
        pytest.skip("pyswisseph not installed")
    global_state.auto_configure()
    grid = geodesic.fibonacci_sphere(60)
    cfg = StreamConfig(window_frames=3, window_stride=3)
    base = parse_datetime("2022-01-01T00:00:00Z")

    # continuous (epoch 0, 24 h stride) over a bounded modern range
    cont = CurriculumStream(grid, cfg, start_jd=base, end_jd=base + 12.0)
    cont.set_epoch(0)
    e, lon = next(iter(cont))
    assert e.shape == (3, 60, C.LOCAL_FIELD_WIDTH) and e.dtype == torch.float32
    assert lon.shape == (3, C.N_BODIES)
    assert torch.isfinite(e).all()

    # micro-burst (epoch 20, 24 s stride) — islice keeps it to a couple windows
    burst = CurriculumStream(grid, cfg, start_jd=base, end_jd=base + 31.0, seed=0)
    burst.set_epoch(20)
    got = list(itertools.islice(iter(burst), 2))
    assert len(got) == 2
    for e2, lon2 in got:
        assert e2.shape == (3, 60, C.LOCAL_FIELD_WIDTH)
        assert lon2.shape == (3, C.N_BODIES)
