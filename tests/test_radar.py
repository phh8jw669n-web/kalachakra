import numpy as np

from kalachakra.analysis import radar


def test_temporal_stride_bounds_points():
    assert radar.temporal_stride(500) == 1
    assert radar.temporal_stride(10_000, target_points=1000) == 10
    assert radar.temporal_stride(13_400_000_000, target_points=1000) > 1


def test_band_gains_attenuate_high_freq_at_large_stride():
    slow = radar.band_gains(stride=1)          # fine inspection
    fast = radar.band_gains(stride=10**7)      # deep-time scrubbing
    # Micro band collapses when scrubbing fast; macro band stays high.
    assert fast["micro"] < 0.5 < slow["micro"]
    assert fast["macro"] > fast["micro"]
    for g in (slow, fast):
        assert all(0.0 <= v <= 1.0 for v in g.values())


def test_significance_percentile_monotonic():
    p1 = radar.significance_percentile(1)
    p100 = radar.significance_percentile(100)
    p10k = radar.significance_percentile(10256)
    assert p1 < p100 < p10k
    assert abs(p1 - 95.0) < 1e-6 and abs(p10k - 99.99) < 1e-6


def test_is_applying_sign():
    # Two bodies 4 deg apart approaching conjunction (0 deg): applying.
    lons = np.zeros(10)
    lons[0] = np.deg2rad(4.0); lons[5] = 0.0
    speeds = np.zeros(10)
    speeds[0] = np.deg2rad(-1.0)   # body 0 moving toward body 5
    assert radar.is_applying(lons, speeds, 0, 5) is True
    speeds[0] = np.deg2rad(+1.0)   # moving away
    assert radar.is_applying(lons, speeds, 0, 5) is False


def test_build_news_card_is_pure_geometry():
    from kalachakra.ephemeris.global_state import encode_body
    rng = np.random.default_rng(0)
    frame = np.stack([
        encode_body(rng.uniform(-np.pi, np.pi), 0.0, 1.0, rng.normal(), 0.0, 0.0)
        for _ in range(10)
    ])
    card = radar.build_news_card(2451545.0, 51.5, -0.12, frame,
                                 macro_id=12, micro_id=5, rarity=0.97,
                                 rarity_percentile=99.9)
    d = card.to_dict()
    assert d["macro"] == 12 and d["micro"] == 5
    assert isinstance(d["applying"], bool)
    # 9 weighted bodies (Ayanamsha excluded), each with a 3-vector.
    assert len(d["bodies"]) == 9
    assert len(d["bodies"][0]["unit_vector"]) == 3
    assert "rarity_percentile" in d
