"""version5: the GPU-native Sky-Energy Autoencoder end to end (12-body upgrade).

Covers the single-query ephemeris bridge (12 bodies + ecliptic/equatorial/obliquity),
the vectorised spherical math (horizon + Ascendant/MC/Vertex), the two-input model
(shapes + OKLab bounds), the Monte-Carlo dataset, training/resume, the ONNX export
parity, and the /telemetry micro-payload.
"""

import json
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import version5  # noqa: F401,E402  (installs the src path shim)
from kalachakra.ephemeris import global_state                    # noqa: E402
from kalachakra.ephemeris.calendar import parse_datetime         # noqa: E402
from version5 import ephemeris as ephem                          # noqa: E402
from version5 import sky_math                                    # noqa: E402
from version5.config import (                                    # noqa: E402
    DataConfig, ModelConfig, TrainConfig, V5Config,
)
from version5.model import SkyEnergyEncoder, build_model         # noqa: E402


def _skip_no_ephem():
    if not global_state.ephemeris_available():
        pytest.skip("pyswisseph not installed")


# ---------------------------------------------------------------------------
# ephemeris bridge + micro-payload (12 bodies)
# ---------------------------------------------------------------------------
def test_ecliptic_state_equatorial_and_telemetry():
    _skip_no_ephem()
    ephem.configure()
    assert ephem.N_BODIES == 12
    assert ephem.BODY_NAMES[-2:] == ("MeanNode", "TrueNode")
    jd = parse_datetime("2026-08-26T19:00:00Z")
    ecl = ephem.ecliptic_state(jd)
    assert ecl.shape == (12, 4) and np.isfinite(ecl).all()
    eps = ephem.obliquity_rad(jd)
    assert math.radians(23.0) < eps < math.radians(24.0)          # obliquity of date
    eq = ephem.ecl_to_equatorial(ecl, eps)
    assert eq.shape == (12, 2)
    # the rotation reproduces pyswisseph's native equatorial output
    flags = global_state._calc_flags() | 2048                     # FLG_EQUATORIAL
    for i, sid in enumerate(ephem.BODY_SWE_IDS):
        v = global_state.swe.calc_ut(float(jd), sid, flags)[0]
        dra = abs((eq[i, 0] - v[0] + 180) % 360 - 180)
        assert dra < 1e-6 and abs(eq[i, 1] - v[1]) < 1e-6

    tel = ephem.telemetry(jd)
    assert set(tel) == {"jd", "gast_hours", "gast_deg", "obliquity_deg", "bodies"}
    assert set(tel["bodies"]) == set(ephem.BODY_NAMES)
    b = tel["bodies"]["Sun"]
    assert {"ra", "dec", "lon", "lat", "dist", "lon_speed"} <= set(b)
    assert len(json.dumps(tel)) < 2048                            # still a micro-payload


# ---------------------------------------------------------------------------
# vectorised spherical math (horizon + high-frequency geographic resolvers)
# ---------------------------------------------------------------------------
def _state(jd):
    ecl = ephem.ecliptic_state(jd)
    eps = ephem.obliquity_rad(jd)
    eq = ephem.ecl_to_equatorial(ecl, eps)
    gast = ephem.gast_radians(jd)
    return ecl, eq, eps, gast


def test_local_features_shapes_and_ranges():
    _skip_no_ephem()
    ephem.configure()
    ecl, eq, eps, gast = _state(parse_datetime("2000-01-01T12:00:00Z"))
    rng = np.random.default_rng(0)
    lat, lon = sky_math.sample_locations(rng, 64)
    feats, obs = sky_math.local_features(ecl, eq, eps, gast, lat, lon)
    assert feats.shape == (64, 12, 6) and feats.dtype == np.float32
    assert obs.shape == (64, 3) and obs.dtype == np.float32
    alt = feats[..., sky_math.COL_ALT]
    assert alt.min() >= -math.pi / 2 - 1e-4 and alt.max() <= math.pi / 2 + 1e-4
    tgt = sky_math.recon_target(feats)
    assert tgt.shape == (64, 12, 4)
    assert np.allclose(tgt[..., 0] ** 2 + tgt[..., 1] ** 2, 1.0, atol=1e-4)


def test_ascendant_midheaven_match_swisseph():
    """The vectorised Asc/MC must match pyswisseph's native swe.houses()."""
    _skip_no_ephem()
    ephem.configure()
    jd = parse_datetime("2026-08-26T19:00:00Z")
    eps = ephem.obliquity_rad(jd)
    gast = ephem.gast_radians(jd)
    for lat, lon in [(51.5, -0.12), (-33.9, 151.2), (0.0, 0.0), (-70.0, 120.0)]:
        _cusps, ascmc = global_state.swe.houses(float(jd), lat, lon, b"A")
        ramc = np.array([[gast + math.radians(lon)]])
        phi = np.array([[math.radians(lat)]])
        asc, mc, _vx = sky_math.ascendant_mc_vertex(ramc, phi, eps)
        da = abs((math.degrees(asc[0, 0]) - ascmc[0] + 180) % 360 - 180)
        dm = abs((math.degrees(mc[0, 0]) - ascmc[1] + 180) % 360 - 180)
        assert da < 1e-4 and dm < 1e-4


def test_north_pole_altitude_equals_declination():
    """At the north pole every body's altitude == its declination — invariant."""
    _skip_no_ephem()
    ephem.configure()
    ecl, eq, eps, gast = _state(parse_datetime("2024-06-21T00:00:00Z"))
    lat = np.array([math.radians(89.9999)])
    lon = np.array([math.radians(123.0)])
    feats, _obs = sky_math.local_features(ecl, eq, eps, gast, lat, lon)
    alt = feats[0, :, sky_math.COL_ALT]
    assert np.max(np.abs(alt - np.deg2rad(eq[:, 1]))) < 1e-3


def test_sample_locations_area_uniform():
    rng = np.random.default_rng(1)
    lat, _lon = sky_math.sample_locations(rng, 20000)
    assert abs(np.mean(np.sin(lat))) < 0.03


def test_random_jd_is_24s_quantized():
    rng = np.random.default_rng(2)
    s, e = parse_datetime("-3101-02-18T00:00:00"), parse_datetime("7155-02-18T00:00:00")
    for _ in range(200):
        jd = sky_math.random_jd_quantized(rng, s, e)
        assert s <= jd < e
        ticks = (jd - s) / sky_math.VIGHATIKA_DAYS
        assert abs(ticks - round(ticks)) < 1e-2


# ---------------------------------------------------------------------------
# model (two inputs: body features + observer anchors)
# ---------------------------------------------------------------------------
def _model(**kw):
    cfg = ModelConfig(d_model=32, nhead=4, num_layers=2, dim_feedforward=64, **kw)
    return build_model(cfg).eval()


def test_model_shapes_and_oklab_bounds():
    m = _model()
    feats, obs = torch.randn(5, 12, 6), torch.randn(5, 3)
    recon_body, recon_obs, oklab = m(feats, obs)
    assert recon_body.shape == (5, 12, 4) and recon_obs.shape == (5, 3, 2)
    assert oklab.shape == (5, 3)
    L, a, b = oklab[:, 0], oklab[:, 1], oklab[:, 2]
    assert (L >= 0).all() and (L <= 1).all()
    assert a.abs().max() <= 1.0 + 1e-6 and b.abs().max() <= 1.0 + 1e-6


def test_encoder_attention_and_wrap_continuity():
    m = _model()
    oklab, attn = m.encode(torch.randn(3, 12, 6), torch.randn(3, 3), return_attention=True)
    assert oklab.shape == (3, 3) and attn.shape == (3, 2, 4, 13, 13)   # 13 tokens
    lo = torch.zeros(1, 12, 6); lo[0, 0, sky_math.COL_AZ] = math.radians(359.9)
    hi = torch.zeros(1, 12, 6); hi[0, 0, sky_math.COL_AZ] = math.radians(0.1)
    el, eh = m.encoder._expand_body(lo), m.encoder._expand_body(hi)
    assert torch.max((el - eh).abs()) < 1e-2


def test_gap_pool_builds():
    m = _model(pool="gap")
    recon_body, recon_obs, oklab = m(torch.randn(2, 12, 6), torch.randn(2, 3))
    assert recon_body.shape == (2, 12, 4) and recon_obs.shape == (2, 3, 2)
    assert oklab.shape == (2, 3)


def test_reconstruction_loss_rebalances_observer():
    """The observer term must scale with obs_weight; equal-body (mass_w off) is the
    default and each body contributes an equal per-token loss."""
    from version5.losses import reconstruction_loss
    torch.manual_seed(0)
    rb, tb = torch.randn(4, 12, 4), torch.randn(4, 12, 4)
    ro, to = torch.randn(4, 3, 2), torch.randn(4, 3, 2)
    # closed-form: L = (sum_body + w*obs) / (12 + w)
    body = ((rb - tb) ** 2).mean(-1).sum(-1)
    obs = ((ro - to) ** 2).mean((-1, -2))
    for w in (0.0, 3.0, 10.0):
        want = ((body + w * obs) / (12 + w)).mean()
        got = reconstruction_loss(rb, tb, ro, to, obs_weight=w)
        assert torch.allclose(got, want, atol=1e-6)
    # a larger obs_weight pulls the total toward the observer's error
    near = reconstruction_loss(rb, tb, ro, to, obs_weight=0.0)
    far = reconstruction_loss(rb, tb, ro, to, obs_weight=50.0)
    assert abs(float(far) - float(obs.mean())) < abs(float(near) - float(obs.mean()))


def test_defaults_disable_mass_and_set_obs_weight():
    from version5.config import TrainConfig
    tc = TrainConfig()
    assert tc.mass_weighting is False and tc.obs_weight == 3.0


def test_mass_weights_len_12():
    from version5.losses import mass_weights
    w = mass_weights()
    assert w.shape == (12,) and float(w[-1]) == 0.5 and float(w[-2]) == 0.5


def test_angular_features_are_sin_cos_pairs():
    """Every cyclic angle (5 body + 3 observer) enters the Transformer as sin/cos,
    never a raw scalar; velocity is the only raw (then tanh-bounded) channel."""
    assert set(sky_math.ANGULAR_COLS) == {sky_math.COL_ALT, sky_math.COL_AZ,
                                          sky_math.COL_LON, sky_math.COL_LAT,
                                          sky_math.COL_HPOS}
    assert sky_math.SCALAR_COLS == (sky_math.COL_VEL,)
    m = _model()
    body = m.encoder._expand_body(torch.zeros(1, 12, 6))          # angles=0
    assert body.shape == (1, 12, 2 * 5 + 1)                       # 5*(sin,cos)+velocity
    # sin(0)=0, cos(0)=1 -> first 5 channels 0, next 5 channels 1
    assert torch.allclose(body[0, 0, :5], torch.zeros(5))
    assert torch.allclose(body[0, 0, 5:10], torch.ones(5))
    obs = m.encoder._expand_observer(torch.zeros(1, 3))
    assert obs.shape == (1, 6)


def test_velocity_is_tanh_bounded():
    """Extreme longitude velocities are squashed strictly into [-1,1]."""
    m = _model()
    feats = torch.zeros(2, 12, 6)
    feats[..., sky_math.COL_VEL] = 1000.0                         # absurd velocity
    body = m.encoder._expand_body(feats)
    vel = body[..., -1]                                           # the velocity channel
    assert vel.abs().max() <= 1.0 + 1e-6 and torch.allclose(vel, torch.ones_like(vel))


def test_cosine_warmup_hits_lr_min_at_final_step():
    from version5.training import cosine_warmup
    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=3e-4)
    sch = cosine_warmup(opt, warmup_steps=1000, max_steps=8000,
                        base_lr=3e-4, lr_min=1e-6)
    lrs = []
    for _ in range(8000):
        lrs.append(opt.param_groups[0]["lr"]); opt.step(); sch.step()
    assert abs(max(lrs) - 3e-4) < 1e-9                            # peaks at base lr
    assert abs(lrs[999] - 3e-4) < 1e-9                            # warmup ends at 1000
    assert abs(lrs[-1] - 1e-6) < 2e-8                             # lands on lr_min


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------
def test_dataset_yields_prebatched():
    _skip_no_ephem()
    ephem.configure()
    from version5.dataset import build_dataloader
    cfg = DataConfig(start_jd=parse_datetime("1990-01-01"),
                     end_jd=parse_datetime("2000-01-01"),
                     locations_per_step=16, seed=3)
    loader = build_dataloader(cfg, num_workers=0)
    feats, obs, target, jd = next(iter(loader))
    assert feats.shape == (16, 12, 6) and obs.shape == (16, 3)
    assert target.shape == (16, 12, 4) and isinstance(jd, float)
    assert torch.isfinite(feats).all() and torch.isfinite(obs).all()


# ---------------------------------------------------------------------------
# training + resume
# ---------------------------------------------------------------------------
def _tiny_cfg(tmp_path, **train_kw):
    data = DataConfig(start_jd=parse_datetime("1995-01-01"),
                      end_jd=parse_datetime("2005-01-01"),
                      locations_per_step=32, seed=0)
    model = ModelConfig(d_model=32, nhead=4, num_layers=2, dim_feedforward=64)
    train = TrainConfig(max_steps=4, warmup_steps=1, save_every=100, log_every=1,
                        amp=False, device="cpu", out_dir=str(tmp_path / "run"),
                        num_workers=0, **train_kw)
    return V5Config(model=model, data=data, train=train)


def test_training_runs_and_resumes(tmp_path):
    _skip_no_ephem()
    from version5.training import load_checkpoint, train
    cfg = _tiny_cfg(tmp_path)
    final = train(cfg, max_steps=4)
    assert final.exists()
    _m, payload, _cfg = load_checkpoint(final, map_location="cpu")
    assert payload["step"] == 4 and np.isfinite(payload["metrics"]["loss"])
    final2 = train(cfg, resume=str(final), max_steps=7)
    assert load_checkpoint(final2, map_location="cpu")[1]["step"] == 7


# ---------------------------------------------------------------------------
# ONNX export parity  (two inputs, client math == server math)
# ---------------------------------------------------------------------------
def test_onnx_export_parity(tmp_path):
    _skip_no_ephem()
    ort = pytest.importorskip("onnxruntime")
    from version5.export_onnx import export
    from version5.training import load_checkpoint, train
    cfg = _tiny_cfg(tmp_path)
    final = train(cfg, max_steps=4)
    out = tmp_path / "model_v5.onnx"
    export(str(final), str(out), verify=False)
    assert out.exists() and (tmp_path / "golden.json").exists()

    _m, _payload, ccfg = load_checkpoint(final, map_location="cpu")
    enc = SkyEnergyEncoder(ccfg.model)
    enc.load_state_dict(_m.encoder.state_dict())
    enc.eval()
    pf, po = torch.randn(29, 12, 6), torch.randn(29, 3)          # unseen batch size
    with torch.no_grad():
        ref = enc(pf, po).numpy()
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    got = sess.run(["oklab"], {"features": pf.numpy(), "observer": po.numpy()})[0]
    assert np.max(np.abs(ref - got)) < 1e-4


# ---------------------------------------------------------------------------
# server micro-payload
# ---------------------------------------------------------------------------
def test_server_telemetry_endpoint():
    _skip_no_ephem()
    pytest.importorskip("fastapi")
    from starlette.testclient import TestClient

    from version5.server import app
    c = TestClient(app)
    r = c.get("/telemetry?time=2026-08-26T19:00:00Z")
    assert r.status_code == 200 and len(r.content) < 2048
    d = r.json()
    assert set(d["bodies"]) == set(ephem.BODY_NAMES) and "obliquity_deg" in d
    assert c.get("/telemetry").status_code == 200
    info = c.get("/api/info").json()
    assert info["service"] == "kalachakra-version5" and len(info["bodies"]) == 12
