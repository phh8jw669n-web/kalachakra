"""version8: the 88-D relational SIREN engine.

Covers the 88-D state (33 local + 55 chords), the balanced isometric loss, the 4x128 SIREN
with its gamut-bounded head, the export→re-run parity that guarantees the JS/GLSL ports match
PyTorch, an altitude cross-check against pyswisseph, and a tiny train→export cycle.
"""

import json
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import version8  # noqa: F401,E402
from version8 import state as st                                # noqa: E402
from version8.config import DataConfig, SirenConfig, TrainConfig, V8Config   # noqa: E402
from version8.ephemeris import BODY_NAMES, topocentric_tensor  # noqa: E402
from version8.losses import balanced_sky_distance, isometric_loss   # noqa: E402
from version8.siren import bound_lab, build_siren               # noqa: E402


# ---------------------------------------------------------------------------
# 88-D state
# ---------------------------------------------------------------------------
def test_state_geometry():
    assert st.N_LOCAL == 33 and st.N_CHORD == 55 and st.STATE_DIM == 88
    assert len(st.PAIRS) == 55
    # canonical i<j order
    assert st.PAIRS[0] == (0, 1) and st.PAIRS[-1] == (9, 10)
    assert all(i < j for i, j in st.PAIRS)


def test_state_shape_units_and_chords():
    rng = np.random.default_rng(0)
    n = 64
    lat, lon = rng.uniform(-90, 90, n), rng.uniform(-180, 180, n)
    jd = rng.uniform(st.__dict__.get("J2000", 2451545.0) - 1e6, 2451545.0 + 1e6, n)
    state = st.topocentric_state(lat, lon, jd)
    assert state.shape == (n, 88) and state.dtype == np.float32 and np.isfinite(state).all()
    local, chords = st.split_state(state)
    # each body triple is a unit vector
    v = local.reshape(n, 11, 3)
    assert np.allclose(np.linalg.norm(v, axis=-1), 1.0, atol=1e-4)
    # chords are dot products of unit vectors -> [-1, 1], and match an independent recompute
    assert chords.min() >= -1.0001 and chords.max() <= 1.0001
    for k, (i, j) in enumerate(st.PAIRS):
        assert np.allclose(chords[:, k], np.sum(v[:, i] * v[:, j], axis=1), atol=1e-5)


def test_altitude_matches_swisseph_near_present():
    swe = pytest.importorskip("swisseph")
    flags = swe.FLG_SWIEPH | swe.FLG_TOPOCTR | swe.FLG_EQUATORIAL
    swe_ids = [swe.SUN, swe.MOON, swe.MERCURY, swe.VENUS, swe.MARS, swe.JUPITER,
               swe.SATURN, swe.URANUS, swe.NEPTUNE, swe.PLUTO]
    pts = [(48.8566, 2.3522, 2451545.0), (-33.8688, 151.2093, 2459000.25),
           (35.6762, 139.6503, 2447000.0), (40.7128, -74.0060, 2458849.5)]
    for lat, lon, jd in pts:
        swe.set_topo(lon, lat, 0.0)
        geopos = (lon, lat, 0.0)
        sky = topocentric_tensor(np.array([lat]), np.array([lon]), np.array([jd]))[0]
        for i, sid in enumerate(swe_ids):
            ra, dec, dist = swe.calc_ut(jd, sid, flags)[0][:3]
            _az, alt, _app = swe.azalt(jd, swe.EQU2HOR, geopos, 0.0, 0.0, (ra, dec, dist))
            alt_ours = math.degrees(math.asin(max(-1.0, min(1.0, float(sky[i * 3 + 2])))))
            tol = 1.6 if sid == swe.MOON else 0.6
            assert abs(alt_ours - alt) < tol, (BODY_NAMES[i], jd, alt_ours, alt)


# ---------------------------------------------------------------------------
# SIREN + gamut head + export parity
# ---------------------------------------------------------------------------
def test_siren_shapes_and_gamut_bounds():
    net = build_siren(hidden=32, hidden_layers=4).eval()
    y = net(torch.randn(9, 88))
    assert y.shape == (9, 3)
    # extreme logits must stay inside the gamut box
    lab = bound_lab(torch.randn(20000, 3) * 40.0, 5.0, 90.0, 80.0)
    assert lab[:, 0].min() >= 5.0 and lab[:, 0].max() <= 95.0      # L* bounded to [5, 95]
    assert lab[:, 1:].abs().max() <= 80.0 + 1e-4                   # a*, b* bounded to [-80, 80]
    assert lab[:, 0].max() > 90.0 and lab[:, 0].min() < 10.0       # and the bounds are actually used


def test_export_structure():
    w = build_siren(hidden=16, hidden_layers=4).export_weights()
    assert w["in_features"] == 88 and w["hidden"] == 16 and w["hidden_layers"] == 4
    assert w["output_activation"] == "v8_gamut"
    assert w["lab_l0"] == 5.0 and w["lab_lspan"] == 90.0 and w["lab_ab"] == 80.0
    assert [ly["activation"] for ly in w["layers"]] == ["sin", "sin", "sin", "sin", "linear"]
    assert len(w["layers"][0]["W"][0]) == 88


def _numpy_siren(weights, x):
    """Dependency-free re-run of exported weights — mirrors siren8.js (makeSiren + boundLab)."""
    omega0 = weights["omega0"]
    h = np.asarray(x, dtype=np.float64)
    for layer in weights["layers"]:
        W = np.asarray(layer["W"])
        b = np.asarray(layer["b"])
        h = h @ W.T + b
        if layer["activation"] == "sin":
            h = np.sin(omega0 * h)
    l0, ls, ab = weights["lab_l0"], weights["lab_lspan"], weights["lab_ab"]
    L = l0 + ls / (1.0 + np.exp(-h[..., 0]))
    a = ab * np.tanh(h[..., 1])
    bb = ab * np.tanh(h[..., 2])
    return np.stack([L, a, bb], axis=-1)


def test_export_matches_torch_forward():
    torch.manual_seed(1)
    net = build_siren(hidden=64, hidden_layers=4).eval()
    w = net.export_weights()
    x = torch.randn(12, 88)
    with torch.no_grad():
        ref = net(x).numpy()
    got = _numpy_siren(w, x.numpy())
    assert np.max(np.abs(ref - got)) < 1e-3     # the JS/GLSL parity contract


# ---------------------------------------------------------------------------
# balanced isometric loss
# ---------------------------------------------------------------------------
def test_balanced_sky_distance_split():
    torch.manual_seed(0)
    state = torch.rand(32, 88)
    d = balanced_sky_distance(state)
    assert d.shape == (32, 32)
    assert torch.diagonal(d).abs().max() < 2e-3       # self-distance ~0 (cdist rounding)
    assert (d >= 0).all()
    assert torch.allclose(d, d.T, atol=1e-5)          # symmetric


def test_chords_carry_no_spatial_signal_so_local_must_dominate():
    """Regression for the near-flat globe: the 55 chords are rotation-invariant, so at a
    fixed instant they are identical for every observer and add ZERO across-globe variation.
    All geographic colour structure therefore lives in the 33 local vectors, and the balanced
    distance must weight them dominantly (the old 0.5/0.5 split washed the globe out)."""
    jd0 = 2451545.0
    lat = np.repeat(np.linspace(-80, 80, 12), 24)
    lon = np.tile(np.linspace(-175, 175, 24), 12)
    state = st.topocentric_state(lat, lon, np.full(lat.shape, jd0))   # fixed time
    _local, chords = st.split_state(state)
    # chords are effectively constant across the whole globe at a fixed instant
    assert chords.std(axis=0).mean() < 1e-4

    stt = torch.from_numpy(state)
    # with the chord weight the fixed-time distance is (numerically) all local
    d_full = balanced_sky_distance(stt)                       # defaults 0.7 / 0.3
    d_local_only = balanced_sky_distance(stt, w_local=1.0, w_chord=0.0)
    assert torch.allclose(d_full, 0.7 * d_local_only, atol=1e-4)
    # the default weighting is local-dominant, so the globe has real contrast to show
    assert float(d_full.max()) > 0.3


def test_isometric_loss_properties():
    torch.manual_seed(0)
    state = torch.rand(48, 88)
    d_sky = balanced_sky_distance(state)
    gamma = 15.0
    # colour whose pairwise distances already equal gamma*d_sky -> near-zero loss.
    # embed the sky distance matrix into 3-D is impossible in general; instead check the
    # two guard cases: collapsed colour is heavily penalised; general case finite & >= 0.
    collapsed = torch.full((48, 3), 0.5)
    assert isometric_loss(state, collapsed, gamma) > 1.0
    loss = isometric_loss(state, torch.randn(48, 3) * 30, gamma)
    assert torch.isfinite(loss) and float(loss) >= 0.0
    _ = d_sky


# ---------------------------------------------------------------------------
# dataset + end to end
# ---------------------------------------------------------------------------
def test_dataset_yields_88d():
    from version8.dataset import build_dataloader
    loader = build_dataloader(DataConfig(batch=16, seed=1), num_workers=0)
    it = iter(loader)
    (a,) = next(it)
    (b,) = next(it)
    assert a.shape == (16, 88) and a.dtype == torch.float32 and torch.isfinite(a).all()
    assert not torch.equal(a, b)


def _tiny_cfg(tmp_path, **tk):
    siren = SirenConfig(hidden=16, hidden_layers=4)
    data = DataConfig(batch=48, seed=0)
    train = TrainConfig(max_steps=4, warmup_steps=1, save_every=100, log_every=1,
                        device="cpu", num_workers=0, out_dir=str(tmp_path / "run"), **tk)
    return V8Config(siren=siren, data=data, train=train)


def test_training_runs_resumes_and_exports(tmp_path):
    from version8.training import CHECKPOINT_FORMAT, export_weights_json, load_checkpoint, train
    cfg = _tiny_cfg(tmp_path)
    final = train(cfg, max_steps=4)
    assert final.exists()
    _m, payload, _c = load_checkpoint(final, map_location="cpu")
    assert payload["step"] == 4 and payload["format"] == CHECKPOINT_FORMAT
    final2 = train(cfg, resume=str(final), max_steps=7)
    assert load_checkpoint(final2, map_location="cpu")[1]["step"] == 7

    out = tmp_path / "weights.json"
    export_weights_json(str(final2), str(out))
    w = json.loads(out.read_text())
    assert w["output_activation"] == "v8_gamut" and "gamma" in w
    assert w["w_local"] == 0.7 and w["w_chord"] == 0.3       # local-dominant provenance
    model, _p, _c = load_checkpoint(final2, map_location="cpu")
    model.eval()
    x = st.topocentric_state(np.array([10.0, -40.0]), np.array([20.0, 100.0]), np.array([2451545.0, 2460000.0]))
    with torch.no_grad():
        ref = model(torch.from_numpy(x)).numpy()
    got = _numpy_siren(w, x)
    assert np.allclose(ref, got, atol=1e-3)
