"""version9: the Topocentric Self-Attention engine.

Covers the geometry (33 local + 55 HORIZON-GATED chords), the observer-dependence that makes
v9 different from v8 (gated chords and attention both vary across the globe at a fixed instant),
the single-head attention model with its gamut-bounded head, the export -> re-run parity that
guarantees the JS/GLSL ports match PyTorch, an altitude cross-check against pyswisseph, and a
tiny train -> export cycle.
"""

import json
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import version9  # noqa: F401,E402
from version9 import state as st                                    # noqa: E402
from version9.attention import bound_ab, build_model                # noqa: E402
from version9.config import AttnConfig, DataConfig, TrainConfig, V9Config   # noqa: E402
from version9.ephemeris import BODY_NAMES, topocentric_tensor       # noqa: E402
from version9.losses import balanced_sky_distance, isometric_loss   # noqa: E402


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def test_state_geometry():
    assert st.N_LOCAL == 33 and st.N_CHORD == 55 and st.STATE_DIM == 88
    assert len(st.PAIRS) == 55
    assert st.PAIRS[0] == (0, 1) and st.PAIRS[-1] == (9, 10)
    assert all(i < j for i, j in st.PAIRS)


def test_local_units_and_gated_chords():
    rng = np.random.default_rng(0)
    n = 64
    lat, lon = rng.uniform(-90, 90, n), rng.uniform(-180, 180, n)
    jd = rng.uniform(2451545.0 - 1e6, 2451545.0 + 1e6, n)
    local = st.local_vectors(lat, lon, jd)
    assert local.shape == (n, 33) and local.dtype == np.float32 and np.isfinite(local).all()
    v = local.reshape(n, 11, 3)
    assert np.allclose(np.linalg.norm(v, axis=-1), 1.0, atol=1e-4)
    # gated chords R_ij = g_i g_j (v_i.v_j); |R| <= 1 and match an independent recompute
    gk = 8.0
    chords = st.gated_chords(local, gk)
    g = 1.0 / (1.0 + np.exp(-gk * v[:, :, 2]))
    for k, (i, j) in enumerate(st.PAIRS):
        want = g[:, i] * g[:, j] * np.sum(v[:, i] * v[:, j], axis=1)
        assert np.allclose(chords[:, k], want, atol=1e-5)
    assert np.abs(chords).max() <= 1.0001


def test_gated_chords_are_observer_dependent():
    """The crux of v9: unlike v8's raw chords (rotation-invariant -> identical everywhere at a
    fixed instant), horizon-gated chords vary strongly across the globe, so they carry real
    geographic signal and the globe is not a flat smear."""
    jd0 = 2451545.0
    lat = np.repeat(np.linspace(-80, 80, 12), 24)
    lon = np.tile(np.linspace(-175, 175, 24), 12)
    local = st.local_vectors(lat, lon, np.full(lat.shape, jd0))
    v = local.reshape(-1, 11, 3)
    raw = np.stack([np.sum(v[:, i] * v[:, j], 1) for i, j in st.PAIRS], 1)   # v8-style
    gated = st.gated_chords(local, 8.0)                                       # v9
    assert raw.std(axis=0).mean() < 1e-4               # raw chords: no spatial signal
    assert gated.std(axis=0).mean() > 0.05             # gated chords: real spatial signal


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
# model + gamut head + export parity
# ---------------------------------------------------------------------------
def test_model_shapes_and_chroma_bounds():
    net = build_model(d_model=16, d_ff=32, d_head=16, n_blocks=2).eval()
    y = net(torch.randn(9, 11, 3))
    assert y.shape == (9, 2)                            # pure a*, b* — no luminance channel
    y2 = net(torch.randn(9, 33))                       # 2-D input is auto-reshaped
    assert y2.shape == (9, 2)
    ab = bound_ab(torch.randn(20000, 2) * 40.0, 80.0)
    assert ab.abs().max() <= 80.0 + 1e-4               # a*, b* bounded to [-80, 80]
    assert ab.max() > 79.0 and ab.min() < -79.0        # and the bounds are actually used


def test_export_structure():
    w = build_model(d_model=16, d_ff=32, d_head=16, n_blocks=2).export_weights()
    assert w["arch"] == "v9_topo_attention" and w["output_activation"] == "v9_chroma"
    assert w["out_features"] == 2 and "lab_l" in w and w["lab_ab"] == 80.0
    assert w["n_bodies"] == 11 and w["token_dim"] == 3 and w["d_model"] == 16
    assert w["d_ff"] == 32 and w["d_head"] == 16 and w["n_blocks"] == 2 and "vis_bias" in w
    assert len(w["blocks"]) == 2 and "tau" in w["blocks"][0]
    assert len(w["W_in"]) == 16 and len(w["W_in"][0]) == 3
    assert len(w["E_body"]) == 11 and len(w["Wo2"]) == 2      # a*, b* only


def _numpy_model(w, local):
    """Dependency-free re-run of exported weights — mirrors attn9.js / shader9.js exactly."""
    def sm(x, axis):
        x = x - x.max(axis=axis, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=axis, keepdims=True)

    x = np.asarray(local, dtype=np.float64).reshape(-1, w["n_bodies"], w["token_dim"])
    D = w["d_model"]
    scale = 1.0 / math.sqrt(D)
    vis = w["vis_bias"] * x[:, :, 2]

    def lin(h, W, b):
        return h @ np.asarray(W).T + np.asarray(b)

    t = x @ np.asarray(w["W_in"]).T + np.asarray(w["b_in"]) + np.asarray(w["E_body"])
    for blk in w["blocks"]:
        q, k, v = lin(t, blk["Wq"], blk["bq"]), lin(t, blk["Wk"], blk["bk"]), lin(t, blk["Wv"], blk["bv"])
        scores = q @ k.transpose(0, 2, 1) * (scale * blk["tau"]) + vis[:, None, :]
        t = t + sm(scores, -1) @ v
        t = t + np.tanh(lin(t, blk["W1"], blk["b1"])) @ np.asarray(blk["W2"]).T + np.asarray(blk["b2"])
    pw = sm((t @ np.asarray(w["q_pool"])) * (scale * w["tau_pool"]) + vis, 1)
    pooled = np.einsum("nb,nbd->nd", pw, t)
    z = np.tanh(pooled @ np.asarray(w["Wo1"]).T + np.asarray(w["bo1"])) @ np.asarray(w["Wo2"]).T + np.asarray(w["bo2"])
    ab = np.stack([w["lab_ab"] * np.tanh(z[:, 0]), w["lab_ab"] * np.tanh(z[:, 1])], axis=-1)   # [N,2]
    return ab, pw


def test_export_matches_torch_forward():
    torch.manual_seed(1)
    net = build_model(d_model=24, d_ff=48, d_head=24, n_blocks=2).eval()
    w = net.export_weights()
    local = st.local_vectors(np.array([10.0, -40.0, 62.0]), np.array([20.0, 100.0, -5.0]),
                             np.array([2451545.0, 2460000.0, 2440000.0]))
    with torch.no_grad():
        ref, refpool = net(torch.from_numpy(local), return_pool=True)
    got, gotpool = _numpy_model(w, local)
    assert np.max(np.abs(ref.numpy() - got)) < 1e-3           # the JS/GLSL colour contract
    assert np.max(np.abs(refpool.numpy() - gotpool)) < 1e-3   # per-body energy weights match


def test_attention_is_observer_dependent_and_visibility_led():
    """The engine's promise: attention (and thus the pooled energy read-out) shifts with the
    observer's horizon, and prominent (above-horizon) bodies get more energy weight."""
    torch.manual_seed(2)
    net = build_model().eval()
    jd0 = 2451545.0
    lats = np.repeat(np.linspace(-80, 80, 10), 20)
    lons = np.tile(np.linspace(-170, 170, 20), 10)
    local = st.local_vectors(lats, lons, np.full(lats.shape, jd0))
    with torch.no_grad():
        _lab, pool = net(torch.from_numpy(local), return_pool=True)
    pool = pool.numpy()
    # pool weights genuinely vary across observers at a fixed instant (not a global constant)
    assert pool.std(axis=0).mean() > 0.01
    # and they follow the visible sky: correlation with body altitude (zenith) is strongly positive
    zen = local.reshape(-1, 11, 3)[:, :, 2]
    corr = np.corrcoef(pool.ravel(), zen.ravel())[0, 1]
    assert corr > 0.3


# ---------------------------------------------------------------------------
# loss
# ---------------------------------------------------------------------------
def test_balanced_sky_distance_split():
    torch.manual_seed(0)
    feat = torch.rand(32, 88)
    d = balanced_sky_distance(feat)
    assert d.shape == (32, 32)
    assert torch.diagonal(d).abs().max() < 2e-3
    assert (d >= 0).all()
    assert torch.allclose(d, d.T, atol=1e-5)


def test_isometric_loss_properties():
    torch.manual_seed(0)
    feat = torch.rand(48, 88)
    collapsed = torch.full((48, 3), 0.5)
    assert isometric_loss(feat, collapsed, 32.0) > 1.0
    loss = isometric_loss(feat, torch.randn(48, 3) * 30, 32.0)
    assert torch.isfinite(loss) and float(loss) >= 0.0


# ---------------------------------------------------------------------------
# dataset + end to end
# ---------------------------------------------------------------------------
def test_dataset_yields_88d():
    from version9.dataset import build_dataloader
    loader = build_dataloader(DataConfig(batch=16, seed=1), gate_k=8.0, num_workers=0)
    it = iter(loader)
    (a,) = next(it)
    (b,) = next(it)
    assert a.shape == (16, 88) and a.dtype == torch.float32 and torch.isfinite(a).all()
    assert not torch.equal(a, b)


def _tiny_cfg(tmp_path, **tk):
    attn = AttnConfig(d_model=16, d_ff=32, d_head=16, n_blocks=2)
    data = DataConfig(batch=48, seed=0)
    train = TrainConfig(max_steps=4, warmup_steps=1, save_every=100, log_every=1,
                        device="cpu", num_workers=0, out_dir=str(tmp_path / "run"), **tk)
    return V9Config(attn=attn, data=data, train=train)


def test_training_runs_resumes_and_exports(tmp_path):
    from version9.training import CHECKPOINT_FORMAT, export_weights_json, load_checkpoint, train
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
    assert w["arch"] == "v9_topo_attention" and "gamma" in w
    assert w["output_activation"] == "v9_chroma" and w["out_features"] == 2
    assert w["w_local"] == 0.5 and w["w_rel"] == 0.5 and "gate_k" in w
    model, _p, _c = load_checkpoint(final2, map_location="cpu")
    model.eval()
    local = st.local_vectors(np.array([10.0, -40.0]), np.array([20.0, 100.0]), np.array([2451545.0, 2460000.0]))
    with torch.no_grad():
        ref = model(torch.from_numpy(local)).numpy()
    got, _pw = _numpy_model(w, local)
    assert np.allclose(ref, got, atol=1e-3)
