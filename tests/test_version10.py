"""version10: the 13-token Astrological Anchor engine (11 bodies + Ascendant + Midheaven).

Covers the token expansion (39 local + 78 gated chords), the ASC/MC astronomy (cross-checked
against pyswisseph houses), their topocentric geometry (ASC on the horizon, MC on the meridian),
the softer horizon gate, the TV smoothness term, the OKLCH head, and the export -> re-run parity
that guarantees the JS/GLSL ports match PyTorch.
"""

import json
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import version10  # noqa: F401,E402
from version10 import state as st                                    # noqa: E402
from version10.attention import bound_cartesian, build_model         # noqa: E402
from version10.config import AttnConfig, DataConfig, TrainConfig, V10Config   # noqa: E402
from version10.ephemeris import BODY_NAMES, N_BODIES, asc_mc_ecliptic, gmst_deg, _obliquity  # noqa: E402
from version10.losses import (   # noqa: E402
    balanced_sky_distance, isometric_loss, isometric_pair_loss, tv_loss,
)

DEG = math.pi / 180.0


# ---------------------------------------------------------------------------
# 13-token geometry
# ---------------------------------------------------------------------------
def test_token_geometry():
    assert N_BODIES == 13 and BODY_NAMES[-2:] == ("ASC", "MC")
    assert st.N_LOCAL == 39 and st.N_CHORD == 78 and st.STATE_DIM == 117
    assert len(st.PAIRS) == 78 and st.PAIRS[0] == (0, 1) and st.PAIRS[-1] == (11, 12)


def test_local_units_and_asc_mc_placement():
    rng = np.random.default_rng(0)
    n = 64
    lat, lon = rng.uniform(-88, 88, n), rng.uniform(-180, 180, n)
    jd = rng.uniform(2451545.0 - 1e6, 2451545.0 + 1e6, n)
    local = st.local_vectors(lat, lon, jd)
    assert local.shape == (n, 39) and local.dtype == np.float32 and np.isfinite(local).all()
    v = local.reshape(n, 13, 3)
    assert np.allclose(np.linalg.norm(v[:, :11], axis=-1), 1.0, atol=1e-4)   # 11 bodies are unit
    # ASC (token 11) rises on the horizon: Up ~ 0 ; MC (token 12) is on the meridian: East ~ 0
    # (fade scales all 3 components equally, so these zero components stay zero)
    assert np.abs(v[:, 11, 2]).max() < 1e-3            # ASC zenith component ~ 0 (on the horizon)
    assert np.abs(v[:, 12, 1]).max() < 1e-3            # MC east component ~ 0 (on the meridian)


def test_anchor_polar_fade():
    """v10.1: ASC/MC are unit through the mid-latitudes (fade == 1) and taper to ~0 at the poles,
    so their wild high-latitude variation cannot inject a polar zipper. The 11 bodies are never
    faded."""
    lat = np.array([0.0, 45.0, 59.0, 75.0, 89.0, -89.0])
    lon = np.full_like(lat, 33.0)
    v = st.local_vectors(lat, lon, np.full_like(lat, 2451545.0)).reshape(-1, 13, 3)
    anc = np.linalg.norm(v[:, 11:], axis=-1)          # [N,2] ASC & MC magnitudes
    assert np.allclose(anc[:3], 1.0, atol=1e-4)        # |lat| <= 60 -> full unit anchors
    assert (anc[4] < 0.05).all() and (anc[5] < 0.05).all()   # |lat| ~ 89 -> faded to ~0
    assert (anc[3] < anc[2]).all()                     # monotone taper across the cap
    assert np.allclose(np.linalg.norm(v[:, :11], axis=-1), 1.0, atol=1e-4)   # bodies untouched


def test_asc_mc_matches_swisseph():
    swe = pytest.importorskip("swisseph")
    pts = [(48.8566, 2.3522, 2451545.0), (-33.8688, 151.2093, 2459000.25),
           (40.7128, -74.0060, 2455000.0), (35.6762, 139.6503, 2447000.0), (60.0, 25.0, 2460000.0)]
    for lat, lon, jd in pts:
        T = (jd - 2451545.0) / 36525.0
        eps = _obliquity(np.array([T]))[0]
        lst = (gmst_deg(np.array([jd]))[0] + lon) * DEG
        lam_asc, lam_mc = asc_mc_ecliptic(np.array([lat * DEG]), np.array([lst]),
                                          np.array([math.cos(eps)]), np.array([math.sin(eps)]))
        _cusps, ascmc = swe.houses(jd, lat, lon, b"P")
        da = (math.degrees(lam_asc[0]) - ascmc[0] + 180) % 360 - 180
        dm = (math.degrees(lam_mc[0]) - ascmc[1] + 180) % 360 - 180
        assert abs(da) < 0.05 and abs(dm) < 0.05, (lat, lon, jd, da, dm)


def test_asc_mc_are_observer_dependent():
    """The whole point of the anchors: unlike the geocentric bodies, ASC/MC differ per observer
    at a fixed instant, so they carry sharp spatial (astrocartography) structure."""
    jd0 = 2451545.0
    lat = np.repeat(np.linspace(-55, 55, 12), 24)     # mid-latitudes (fade == 1, no confound)
    lon = np.tile(np.linspace(-175, 175, 24), 12)
    v = st.local_vectors(lat, lon, np.full(lat.shape, jd0)).reshape(-1, 13, 3)
    assert v[:, 0].std(axis=0).max() > 0.05           # bodies vary across the globe too
    assert v[:, 11].std(axis=0).max() > 0.3           # ASC varies strongly with geography
    assert v[:, 12].std(axis=0).max() > 0.3           # MC too


def test_gated_chords_shape_and_signal():
    jd0 = 2451545.0
    lat = np.repeat(np.linspace(-80, 80, 12), 24)
    lon = np.tile(np.linspace(-175, 175, 24), 12)
    local = st.local_vectors(lat, lon, np.full(lat.shape, jd0))
    ch = st.gated_chords(local, 3.0)
    assert ch.shape == (len(lat), 78) and np.abs(ch).max() <= 1.0001
    assert ch.std(axis=0).mean() > 0.02               # gated chords carry across-globe signal


# ---------------------------------------------------------------------------
# model + head + loss
# ---------------------------------------------------------------------------
def test_model_shapes_and_oklch_bounds():
    net = build_model(d_model=16, d_ff=32, d_head=16, n_blocks=2).eval()
    assert net.cfg["n_bodies"] == 13
    y = net(torch.randn(9, 13, 3))
    assert y.shape == (9, 2)
    y2 = net(torch.randn(9, 39))
    assert y2.shape == (9, 2)
    # pure-Cartesian disk head: |(a,b)| < cmax always, approaches cmax for large logits, and has
    # NO hue angle (winding is not representable).
    ab = bound_cartesian(torch.randn(40000, 2) * 8.0, 0.4)
    assert ab.norm(dim=1).max() < 0.4 and ab.norm(dim=1).max() > 0.39
    assert bound_cartesian(torch.zeros(1, 2), 0.4).abs().max() == 0.0      # origin -> 0


def test_export_structure():
    w = build_model(d_model=16, d_ff=32, d_head=16, n_blocks=2).export_weights()
    assert w["arch"] == "v10_topo_attention" and w["output_activation"] == "v10_cartesian"
    assert w["n_bodies"] == 13 and w["out_features"] == 2 and w["okl_cmax"] == 0.4
    assert w["n_anchors"] == 2                                          # ASC/MC exempt from vis prior
    assert w["qk_norm"] is True                                         # v10.1 bounded cosine attn
    assert all(b["tau"] <= 30.0 + 1e-6 for b in w["blocks"]) and w["tau_pool"] <= 30.0 + 1e-6
    assert len(w["E_body"]) == 13 and len(w["Wo2"]) == 2


def _numpy_model(w, local):
    def sm(x, ax):
        x = x - x.max(ax, keepdims=True)
        e = np.exp(x)
        return e / e.sum(ax, keepdims=True)

    def nrm(a, ax):                                # L2-normalise (cosine attention, v10.1)
        return a / np.maximum(np.linalg.norm(a, axis=ax, keepdims=True), 1e-12)

    qk = bool(w.get("qk_norm", False))
    x = np.asarray(local, dtype=np.float64).reshape(-1, w["n_bodies"], w["token_dim"])
    D = w["d_model"]
    s = 1.0 / math.sqrt(D)
    zen = x[:, :, 2].copy()
    na = w.get("n_anchors", 0)
    if na > 0:                                    # ASC/MC are structural axes -> always fully visible
        zen[:, -na:] = 1.0
    vis = w["vis_bias"] * zen

    def lin(h, W, b):
        return h @ np.asarray(W).T + np.asarray(b)

    t = x @ np.asarray(w["W_in"]).T + np.asarray(w["b_in"]) + np.asarray(w["E_body"])
    for bk in w["blocks"]:
        q, k, v = lin(t, bk["Wq"], bk["bq"]), lin(t, bk["Wk"], bk["bk"]), lin(t, bk["Wv"], bk["bv"])
        if qk:                                    # bounded cosine attention: tau is the temperature
            sco = nrm(q, -1) @ nrm(k, -1).transpose(0, 2, 1) * bk["tau"] + vis[:, None, :]
        else:
            sco = q @ k.transpose(0, 2, 1) * (s * bk["tau"]) + vis[:, None, :]
        t = t + sm(sco, -1) @ v
        t = t + np.tanh(lin(t, bk["W1"], bk["b1"])) @ np.asarray(bk["W2"]).T + np.asarray(bk["b2"])
    if qk:
        psco = (nrm(t, -1) @ nrm(np.asarray(w["q_pool"]), 0)) * w["tau_pool"] + vis
    else:
        psco = (t @ np.asarray(w["q_pool"])) * (s * w["tau_pool"]) + vis
    pw = sm(psco, 1)
    pooled = np.einsum("nb,nbd->nd", pw, t)
    z = np.tanh(pooled @ np.asarray(w["Wo1"]).T + np.asarray(w["bo1"])) @ np.asarray(w["Wo2"]).T + np.asarray(w["bo2"])
    if w.get("output_activation") == "v10_cartesian":          # pure-Cartesian disk head (v10.1)
        return w["okl_cmax"] * z / np.sqrt(1.0 + (z ** 2).sum(-1, keepdims=True))
    C = w["okl_cmax"] / (1.0 + np.exp(-z[:, 0]))               # legacy polar OKLCH
    return np.stack([C * np.cos(z[:, 1]), C * np.sin(z[:, 1])], axis=-1)


def test_export_matches_torch_forward():
    torch.manual_seed(1)
    net = build_model(d_model=24, d_ff=48, d_head=24, n_blocks=2).eval()
    w = net.export_weights()
    local = st.local_vectors(np.array([10.0, -40.0, 62.0]), np.array([20.0, 100.0, -5.0]),
                             np.array([2451545.0, 2460000.0, 2440000.0]))
    with torch.no_grad():
        ref = net(torch.from_numpy(local)).numpy()
    assert np.max(np.abs(ref - _numpy_model(w, local))) < 1e-3


def test_tv_and_isometric_loss():
    torch.manual_seed(0)
    feat = torch.rand(48, 117)
    assert balanced_sky_distance(feat).shape == (48, 48)
    assert torch.isfinite(isometric_loss(feat, torch.randn(48, 2) * 0.2, 0.35))
    c = torch.randn(20, 2) * 0.1
    assert float(tv_loss(c, c)) == 0.0                                  # identical -> zero
    assert float(tv_loss(c, c + 0.1)) > 0.0                             # a shift is penalised
    # v10.1 anti-winding: isometry-referenced pair loss is 0 when colour gap == gamma*d_sky (here
    # identical colour AND identical sky), and > 0 for a colour gap with no matching sky change.
    fa = torch.rand(20, 117)
    assert float(isometric_pair_loss(c, c, fa, fa, 0.35)) == 0.0
    assert float(isometric_pair_loss(c, c + 0.3, fa, fa, 0.35)) > 0.0   # winding penalised


# ---------------------------------------------------------------------------
# dataset + end to end
# ---------------------------------------------------------------------------
def test_dataset_yields_117d():
    from version10.dataset import build_dataloader
    loader = build_dataloader(DataConfig(batch=16, seed=1), gate_k=3.0, num_workers=0)
    (a,) = next(iter(loader))
    assert a.shape == (16, 117) and a.dtype == torch.float32 and torch.isfinite(a).all()


def _tiny_cfg(tmp_path, **tk):
    attn = AttnConfig(d_model=16, d_ff=32, d_head=16, n_blocks=2)
    data = DataConfig(batch=48, seed=0)
    train = TrainConfig(max_steps=4, warmup_steps=1, save_every=100, log_every=1,
                        device="cpu", num_workers=0, out_dir=str(tmp_path / "run"), **tk)
    return V10Config(attn=attn, data=data, train=train)


def test_training_runs_resumes_and_exports(tmp_path):
    from version10.training import CHECKPOINT_FORMAT, export_weights_json, load_checkpoint, train
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
    assert w["arch"] == "v10_topo_attention" and w["out_features"] == 2
    assert w["gate_k"] == 3.0 and "tv_weight" in w
    assert w["qk_norm"] is True and "weight_decay" in w and "tv_weight_coarse" in w
    model, _p, _c = load_checkpoint(final2, map_location="cpu")
    model.eval()
    local = st.local_vectors(np.array([10.0, -40.0]), np.array([20.0, 100.0]), np.array([2451545.0, 2460000.0]))
    with torch.no_grad():
        ref = model(torch.from_numpy(local)).numpy()
    assert np.allclose(ref, _numpy_model(w, local), atol=1e-3)
