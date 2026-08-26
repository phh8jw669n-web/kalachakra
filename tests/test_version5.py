"""version5: the GPU-native Sky-Energy Autoencoder end to end.

Covers the single-query ephemeris bridge, the vectorised spherical math, the model
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
# ephemeris bridge + micro-payload
# ---------------------------------------------------------------------------
def test_equatorial_state_and_telemetry():
    _skip_no_ephem()
    ephem.configure()
    jd = parse_datetime("2026-08-26T19:00:00Z")
    eq = ephem.equatorial_state(jd)
    assert eq.shape == (10, 4) and np.isfinite(eq).all()
    assert 0.0 <= eq[0, 0] < 360.0 and -90.0 <= eq[0, 1] <= 90.0     # RA/Dec ranges

    tel = ephem.telemetry(jd)
    assert set(tel) == {"jd", "gast_hours", "gast_deg", "bodies"}
    assert set(tel["bodies"]) == set(ephem.BODY_NAMES)
    # the whole point: a micro-payload well under 2 KB
    assert len(json.dumps(tel)) < 2048


# ---------------------------------------------------------------------------
# vectorised spherical math
# ---------------------------------------------------------------------------
def test_local_features_shapes_and_ranges():
    _skip_no_ephem()
    ephem.configure()
    jd = parse_datetime("2000-01-01T12:00:00Z")
    eq = ephem.equatorial_state(jd)
    gast = ephem.gast_radians(jd)
    rng = np.random.default_rng(0)
    lat, lon = sky_math.sample_locations(rng, 64)
    feats = sky_math.local_features(eq, gast, lat, lon)
    assert feats.shape == (64, 10, 5) and feats.dtype == np.float32
    alt = feats[..., sky_math.COL_ALT]
    assert alt.min() >= -math.pi / 2 - 1e-4 and alt.max() <= math.pi / 2 + 1e-4
    tgt = sky_math.recon_target(feats)
    assert tgt.shape == (64, 10, 4)
    # (sin,cos) pairs are unit vectors
    assert np.allclose(tgt[..., 0] ** 2 + tgt[..., 1] ** 2, 1.0, atol=1e-4)
    assert np.allclose(tgt[..., 2] ** 2 + tgt[..., 3] ** 2, 1.0, atol=1e-4)


def test_north_pole_altitude_equals_declination():
    """At the north pole every body's altitude == its declination — a clean,
    time/longitude-independent invariant that validates the horizon formula."""
    _skip_no_ephem()
    ephem.configure()
    jd = parse_datetime("2024-06-21T00:00:00Z")
    eq = ephem.equatorial_state(jd)
    gast = ephem.gast_radians(jd)
    lat = np.array([math.radians(89.9999)])
    lon = np.array([math.radians(123.0)])           # any longitude
    feats = sky_math.local_features(eq, gast, lat, lon)
    alt = feats[0, :, sky_math.COL_ALT]
    dec = np.deg2rad(eq[:, 1])
    assert np.max(np.abs(alt - dec)) < 1e-3


def test_sample_locations_area_uniform():
    rng = np.random.default_rng(1)
    lat, _lon = sky_math.sample_locations(rng, 20000)
    # sin(lat) ~ U(-1,1): mean of sin(lat) near 0
    assert abs(np.mean(np.sin(lat))) < 0.03


def test_random_jd_is_24s_quantized():
    rng = np.random.default_rng(2)
    s, e = parse_datetime("-3101-02-18T00:00:00"), parse_datetime("7155-02-18T00:00:00")
    for _ in range(200):
        jd = sky_math.random_jd_quantized(rng, s, e)
        assert s <= jd < e
        ticks = (jd - s) / sky_math.VIGHATIKA_DAYS
        # on the 24-second grid (residual is float64 cancellation of two ~4e6 JDs,
        # not a random offset — an unquantised draw would sit ~0.25 off on average)
        assert abs(ticks - round(ticks)) < 1e-2


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
def _model(**kw):
    cfg = ModelConfig(d_model=32, nhead=4, num_layers=2, dim_feedforward=64, **kw)
    return build_model(cfg).eval()


def test_model_shapes_and_oklab_bounds():
    m = _model()
    x = torch.randn(5, 10, 5)
    recon, oklab = m(x)
    assert recon.shape == (5, 10, 4) and oklab.shape == (5, 3)
    L, a, b = oklab[:, 0], oklab[:, 1], oklab[:, 2]
    assert (L >= 0).all() and (L <= 1).all()                       # sigmoid
    assert a.abs().max() <= 1.0 + 1e-6 and b.abs().max() <= 1.0 + 1e-6   # tanh


def test_encoder_attention_and_wrap_continuity():
    m = _model()
    oklab, attn = m.encode(torch.randn(3, 10, 5), return_attention=True)
    assert oklab.shape == (3, 3) and attn.shape == (3, 2, 4, 11, 11)
    # feeding az = 359.9deg vs 0.1deg must not jump (sin/cos expansion)
    lo = torch.zeros(1, 10, 5); lo[0, 0, sky_math.COL_AZ] = math.radians(359.9)
    hi = torch.zeros(1, 10, 5); hi[0, 0, sky_math.COL_AZ] = math.radians(0.1)
    el, eh = m.encoder._expand_angles(lo), m.encoder._expand_angles(hi)
    assert torch.max((el - eh).abs()) < 1e-2


def test_gap_pool_builds():
    m = _model(pool="gap")
    recon, oklab = m(torch.randn(2, 10, 5))
    assert recon.shape == (2, 10, 4) and oklab.shape == (2, 3)


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
    feats, target, jd = next(iter(loader))
    assert feats.shape == (16, 10, 5) and target.shape == (16, 10, 4)
    assert torch.isfinite(feats).all() and isinstance(jd, float)


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
# ONNX export parity  (client math == server math)
# ---------------------------------------------------------------------------
def test_onnx_export_parity(tmp_path):
    _skip_no_ephem()
    ort = pytest.importorskip("onnxruntime")
    from version5.export_onnx import export
    from version5.training import train
    cfg = _tiny_cfg(tmp_path)
    final = train(cfg, max_steps=4)
    out = tmp_path / "model_v5.onnx"
    export(str(final), str(out), verify=False)
    assert out.exists() and (tmp_path / "golden.json").exists()

    # encoder-only ONNX reproduces PyTorch at an unseen batch size (dynamic axis)
    _m, _payload, ccfg = __import__("version5.training", fromlist=["load_checkpoint"]) \
        .load_checkpoint(final, map_location="cpu")
    enc = SkyEnergyEncoder(ccfg.model)
    enc.load_state_dict(_m.encoder.state_dict())
    enc.eval()
    probe = torch.randn(29, 10, 5)
    with torch.no_grad():
        ref = enc(probe).numpy()
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    got = sess.run(["oklab"], {"features": probe.numpy()})[0]
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
    assert set(d["bodies"]) == set(ephem.BODY_NAMES) and "gast_deg" in d
    assert c.get("/telemetry").status_code == 200          # defaults to now
    info = c.get("/api/info").json()
    assert info["service"] == "kalachakra-version5"
