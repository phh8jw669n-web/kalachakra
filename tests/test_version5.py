"""version5.1: the Zero-Redundancy metric-learning encoder, end to end.

Covers the ephemeris bridge + telemetry (still 12 bodies for the frontend), the 50-D
state builder, the Ascendant/MC resolvers, the encoder (state -> OKLab), the isometric
distance-preserving loss, the Monte-Carlo dataset, training/resume, ONNX parity, and
the /telemetry micro-payload.
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
    N_ML_BODIES, STATE_DIM, DataConfig, ModelConfig, TrainConfig, V5Config,
)
from version5.losses import isometric_loss                       # noqa: E402
from version5.model import build_model                           # noqa: E402


def _skip_no_ephem():
    if not global_state.ephemeris_available():
        pytest.skip("pyswisseph not installed")


# ---------------------------------------------------------------------------
# ephemeris bridge + micro-payload (still 12 bodies for the frontend orbits/glow)
# ---------------------------------------------------------------------------
def test_ecliptic_state_equatorial_and_telemetry():
    _skip_no_ephem()
    ephem.configure()
    assert ephem.N_BODIES == 12
    jd = parse_datetime("2026-08-26T19:00:00Z")
    ecl = ephem.ecliptic_state(jd)
    assert ecl.shape == (12, 4) and np.isfinite(ecl).all()
    eps = ephem.obliquity_rad(jd)
    assert math.radians(23.0) < eps < math.radians(24.0)
    eq = ephem.ecl_to_equatorial(ecl, eps)
    assert eq.shape == (12, 2)
    flags = global_state._calc_flags() | 2048                     # FLG_EQUATORIAL
    for i, sid in enumerate(ephem.BODY_SWE_IDS):
        v = global_state.swe.calc_ut(float(jd), sid, flags)[0]
        assert abs((eq[i, 0] - v[0] + 180) % 360 - 180) < 1e-6 and abs(eq[i, 1] - v[1]) < 1e-6

    tel = ephem.telemetry(jd)
    assert set(tel["bodies"]) == set(ephem.BODY_NAMES)
    assert {"ra", "dec", "lon", "lat", "dist", "lon_speed"} <= set(tel["bodies"]["Sun"])
    assert len(json.dumps(tel)) < 2048


# ---------------------------------------------------------------------------
# Zero-Redundancy 50-D state
# ---------------------------------------------------------------------------
def test_config_geometry():
    assert STATE_DIM == 50 and N_ML_BODIES == 11
    assert ModelConfig().state_dim == 50 and ModelConfig().n_bodies == 11
    # the Mean Node (index 10) is dropped; the True Node (index 11) is kept
    assert sky_math.ML_BODY_INDICES == (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11)


def test_local_state_is_cartesian_and_bounded():
    _skip_no_ephem()
    ephem.configure()
    jd = parse_datetime("2000-01-01T12:00:00Z")
    ecl, eps, gast = ephem.ecliptic_state(jd), ephem.obliquity_rad(jd), ephem.gast_radians(jd)
    rng = np.random.default_rng(0)
    lat, lon = sky_math.sample_locations(rng, 64)
    state = sky_math.local_state(ecl, eps, gast, lat, lon)
    assert state.shape == (64, 50) and state.dtype == np.float32
    bodies = state[:, :44].reshape(64, 11, 4)
    # each body's (X,Y,Z) is a unit vector; V is tanh-bounded
    assert np.allclose((bodies[..., :3] ** 2).sum(-1), 1.0, atol=1e-4)
    assert np.abs(bodies[..., 3]).max() <= 1.0
    # observer anchors: unit in the ecliptic plane, Z == 0
    obs = state[:, 44:]
    assert np.allclose(obs[:, 0] ** 2 + obs[:, 1] ** 2, 1.0, atol=1e-4) and np.allclose(obs[:, 2], 0)
    assert np.allclose(obs[:, 3] ** 2 + obs[:, 4] ** 2, 1.0, atol=1e-4) and np.allclose(obs[:, 5], 0)


def test_ascendant_midheaven_match_swisseph():
    _skip_no_ephem()
    ephem.configure()
    jd = parse_datetime("2026-08-26T19:00:00Z")
    eps, gast = ephem.obliquity_rad(jd), ephem.gast_radians(jd)
    for lat, lon in [(51.5, -0.12), (-33.9, 151.2), (0.0, 0.0), (-70.0, 120.0)]:
        _cusps, ascmc = global_state.swe.houses(float(jd), lat, lon, b"A")
        ramc = np.array([[gast + math.radians(lon)]])
        phi = np.array([[math.radians(lat)]])
        asc, mc, _vx = sky_math.ascendant_mc_vertex(ramc, phi, eps)
        assert abs((math.degrees(asc[0, 0]) - ascmc[0] + 180) % 360 - 180) < 1e-4
        assert abs((math.degrees(mc[0, 0]) - ascmc[1] + 180) % 360 - 180) < 1e-4


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
# encoder (state [N,50] -> OKLab [N,3])
# ---------------------------------------------------------------------------
def _model(**kw):
    cfg = ModelConfig(d_model=32, nhead=4, num_layers=2, dim_feedforward=64, **kw)
    return build_model(cfg).eval()


def test_model_shapes_and_oklab_bounds():
    m = _model()
    oklab = m(torch.randn(5, 50))
    assert oklab.shape == (5, 3)
    L, a, b = oklab[:, 0], oklab[:, 1], oklab[:, 2]
    assert (L >= 0).all() and (L <= 1).all()
    assert a.abs().max() <= 1.0 + 1e-6 and b.abs().max() <= 1.0 + 1e-6


def test_encoder_attention_tokens():
    m = _model()
    oklab, attn = m(torch.randn(3, 50), return_attention=True)
    assert oklab.shape == (3, 3) and attn.shape == (3, 2, 4, 12, 12)   # 11 bodies + observer


def test_gap_pool_builds():
    m = _model(pool="gap")
    assert m(torch.randn(2, 50)).shape == (2, 3)


# ---------------------------------------------------------------------------
# isometric distance-preserving loss
# ---------------------------------------------------------------------------
def test_isometric_loss_properties():
    torch.manual_seed(0)
    s3 = torch.randn(64, 3)
    # a perfect isometry (colour is a scaled copy) -> normalised distance matrices match
    assert isometric_loss(s3, 0.3 * s3) < 1e-5
    # a collapsed (constant) colour is heavily penalised
    assert isometric_loss(s3, torch.full((64, 3), 0.5)) > 0.05
    # general case is finite and non-negative; normalised matrices bound the loss to [0,1]
    loss = isometric_loss(torch.randn(64, 50), torch.randn(64, 3))
    assert torch.isfinite(loss) and 0.0 <= float(loss) <= 1.0


def test_cosine_warmup_hits_lr_min_at_final_step():
    from version5.training import cosine_warmup
    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=3e-4)
    sch = cosine_warmup(opt, warmup_steps=1000, max_steps=8000, base_lr=3e-4, lr_min=1e-6)
    lrs = []
    for _ in range(8000):
        lrs.append(opt.param_groups[0]["lr"]); opt.step(); sch.step()
    assert abs(max(lrs) - 3e-4) < 1e-9 and abs(lrs[999] - 3e-4) < 1e-9
    assert abs(lrs[-1] - 1e-6) < 2e-8


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------
def test_dataset_yields_state():
    _skip_no_ephem()
    ephem.configure()
    from version5.dataset import build_dataloader
    cfg = DataConfig(start_jd=parse_datetime("1990-01-01"),
                     end_jd=parse_datetime("2000-01-01"),
                     locations_per_step=16, seed=3)
    loader = build_dataloader(cfg, num_workers=0)
    state, jd = next(iter(loader))
    assert state.shape == (16, 50) and isinstance(jd, float) and torch.isfinite(state).all()


# ---------------------------------------------------------------------------
# training + resume, ONNX parity
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

    enc, _payload, _ccfg = load_checkpoint(final, map_location="cpu")   # model IS the encoder
    enc.eval()
    probe = torch.randn(29, 50)                                          # unseen batch size
    with torch.no_grad():
        ref = enc(probe).numpy()
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    got = sess.run(["oklab"], {"state": probe.numpy()})[0]
    assert np.max(np.abs(ref - got)) < 1e-4


# ---------------------------------------------------------------------------
# server micro-payload (unchanged: 12 bodies for orbits/glow)
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
    assert c.get("/api/info").json()["service"] == "kalachakra-version5"
