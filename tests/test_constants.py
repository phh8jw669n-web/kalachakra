import kalachakra.constants as C


def test_state_widths():
    assert C.GLOBAL_STATE_WIDTH == 70
    assert C.LOCAL_FIELD_WIDTH == 50
    assert C.N_BODIES == 10


def test_memory_partition_sums_to_unified_memory():
    assert sum(C.memory_partition_gb().values()) == C.UNIFIED_MEMORY_GB == 128


def test_frame_count_matches_blueprint_order_of_magnitude():
    frames = C.total_temporal_frames()
    # Blueprint §3.1 states ~13.4 billion frames.
    assert 13.3e9 < frames < 13.6e9


def test_horizon_advance_is_about_one_tenth_degree():
    assert abs(C.HORIZON_ADVANCE_DEG_PER_FRAME - 0.1) < 1e-3


def test_timeline_bounds_start_at_kali_yuga_epoch():
    b = C.timeline_bounds()
    assert b.start_jd == C.KALI_YUGA_EPOCH_JD == 588465.5
    assert b.span_days > 3.7e6  # ~10,256 years
