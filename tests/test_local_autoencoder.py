"""Local Sky Autoencoder (train_v4): physics engine, model, loss, training, inference."""

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from kalachakra.ephemeris import global_state                          # noqa: E402
from kalachakra.ephemeris.calendar import parse_datetime              # noqa: E402
from kalachakra.local_autoencoder import features as F                # noqa: E402
from kalachakra.local_autoencoder.color import oklab_to_srgb8         # noqa: E402
from kalachakra.local_autoencoder.config import (                     # noqa: E402
    DataConfig, LocalSkyConfig, ModelConfig, TrainConfig,
)
from kalachakra.local_autoencoder.losses import (                     # noqa: E402
    oklab_stats, physics_weighted_mse,
)
from kalachakra.local_autoencoder.model import build_model            # noqa: E402


def _skip_no_ephem():
    if not global_state.ephemeris_available():
        pytest.skip("pyswisseph not installed")


# ---------------------------------------------------------------------------
# physics engine
# ---------------------------------------------------------------------------
def test_local_sky_matrix_shapes_and_eclipse():
    _skip_no_ephem()
    global_state.auto_configure()
    jd = parse_datetime("2024-04-08T18:17:00Z")     # a real total solar eclipse
    feat, dist = F.local_sky_matrix(jd, 51.5, -0.12)
    assert feat.shape == (10, 8) and feat.dtype == np.float32
    assert np.isfinite(feat).all()
    # Sun & Moon share the sky at the eclipse -> near-equal azimuth & altitude
    assert abs(feat[0, F.COL_AZ] - feat[1, F.COL_AZ]) < 0.05
    assert abs(feat[0, F.COL_ALT] - feat[1, F.COL_ALT]) < 0.05
    # Moon is the closest body, Sun ~1 AU
    assert dist[1] < 0.01 and 0.98 < dist[0] < 1.02


def test_sample_tensors_and_weights():
    _skip_no_ephem()
    global_state.auto_configure()
    jd = parse_datetime("2000-06-21T12:00:00Z")
    feat, target, weight = F.sample_tensors(jd, 40.0, -74.0)
    assert feat.shape == (10, 8) and target.shape == (11, 8) and weight.shape == (11, 8)
    assert np.allclose(target[:10], feat)                  # rows 0-9 == the bodies
    assert (np.abs(target[10]) <= 1.0 + 1e-6).all()        # observer row in [-1,1]
    assert (weight > 0).all()
    # Moon (close) outweighs Pluto (far) on the same feature column
    assert weight[1, F.COL_AZ] > weight[9, F.COL_AZ]


def test_sphere_sampling_area_uniform():
    rng = np.random.default_rng(0)
    lats = np.array([F.sample_sphere(rng)[0] for _ in range(4000)])
    assert -90 <= lats.min() and lats.max() <= 90
    assert abs(np.mean(np.sin(np.deg2rad(lats)))) < 0.05   # sin(lat) ~ U(-1,1)


# ---------------------------------------------------------------------------
# loss: physics-weighted, wrap-safe
# ---------------------------------------------------------------------------
def test_physics_weighted_mse_wrap_safe():
    target = torch.zeros(1, 11, 8)
    weight = torch.ones(1, 11, 8)
    assert physics_weighted_mse(target.clone(), target, weight) == 0.0
    # a full 2*pi shift on the azimuth (wrap) column costs ~0
    r = target.clone()
    r[0, 0, F.COL_AZ] = 2 * math.pi
    wrap_loss = physics_weighted_mse(r, target, weight)
    assert wrap_loss < 1e-6
    # a real 90-degree azimuth error costs (2 - 2cos(pi/2))/88 = 2/88 -> much bigger
    r2 = target.clone()
    r2[0, 0, F.COL_AZ] = math.pi / 2
    real_loss = physics_weighted_mse(r2, target, weight)
    assert real_loss > 0.02 and real_loss > wrap_loss + 0.01
    # a non-wrap column uses plain squared error
    r3 = target.clone()
    r3[0, 0, F.COL_ANG_VEL] = 1.0
    assert physics_weighted_mse(r3, target, weight) > 0


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
def _model(**kw):
    cfg = ModelConfig(d_model=32, nhead=8, num_layers=2, dim_feedforward=64, **kw)
    return build_model(cfg).eval()


def test_model_shapes_and_oklab_bounds():
    m = _model()
    x = torch.randn(4, 10, 8)
    recon, oklab = m(x)
    assert recon.shape == (4, 11, 8) and oklab.shape == (4, 3)
    L, a, b = oklab[:, 0], oklab[:, 1], oklab[:, 2]
    assert (L >= 0).all() and (L <= 1).all()
    assert a.abs().max() <= 0.5 + 1e-6 and b.abs().max() <= 0.5 + 1e-6


def test_encoder_attention_shape():
    m = _model()
    x = torch.randn(3, 10, 8)
    oklab, attn = m.encode(x, return_attention=True)
    assert oklab.shape == (3, 3)
    assert attn.shape == (3, 2, 8, 11, 11)          # (B, layers, heads, tokens, tokens)


def test_angle_expansion_is_wrap_continuous():
    m = _model()
    lo = torch.zeros(1, 10, 8)
    lo[0, 0, F.COL_AZ] = math.radians(359.9)
    hi = torch.zeros(1, 10, 8)
    hi[0, 0, F.COL_AZ] = math.radians(0.1)
    el, eh = m._expand_angles(lo), m._expand_angles(hi)
    assert torch.max((el - eh).abs()) < 1e-2         # no 360->0 discontinuity


def test_gap_pooling_builds():
    m = _model(pool="gap")
    recon, oklab = m(torch.randn(2, 10, 8))
    assert recon.shape == (2, 11, 8) and oklab.shape == (2, 3)


def test_oklab_stats_and_color():
    st = oklab_stats(torch.tensor([[0.5, 0.2, -0.1], [0.6, -0.2, 0.1]]))
    assert set(st) == {"mean_L", "std_L", "mean_chroma", "mean_abs_a", "mean_abs_b"}
    assert st["mean_chroma"] > 0
    white = oklab_to_srgb8(np.array([1.0, 0.0, 0.0]))
    black = oklab_to_srgb8(np.array([0.0, 0.0, 0.0]))
    assert (white >= 250).all() and (black == 0).all()


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------
def test_dataset_yields_batches():
    _skip_no_ephem()
    global_state.auto_configure()
    from kalachakra.local_autoencoder.dataset import build_dataloader
    cfg = DataConfig(start_jd=parse_datetime("2000-01-01T00:00:00Z"),
                     end_jd=parse_datetime("2001-01-01T00:00:00Z"), seed=1)
    loader = build_dataloader(cfg, batch_size=4, num_workers=0)
    feats, target, weight = next(iter(loader))
    assert feats.shape == (4, 10, 8) and target.shape == (4, 11, 8)
    assert weight.shape == (4, 11, 8) and torch.isfinite(feats).all()


# ---------------------------------------------------------------------------
# training + inference
# ---------------------------------------------------------------------------
def _tiny_cfg(tmp_path, **train_kw):
    data = DataConfig(start_jd=parse_datetime("2000-01-01T00:00:00Z"),
                      end_jd=parse_datetime("2002-01-01T00:00:00Z"))
    model = ModelConfig(d_model=32, nhead=8, num_layers=2, dim_feedforward=64)
    train = TrainConfig(batch_size=8, max_steps=3, warmup_steps=1, save_every=2,
                        amp=False, device="cpu", out_dir=str(tmp_path / "run"),
                        log_every=1, **train_kw)
    return LocalSkyConfig(model=model, data=data, train=train)


def test_training_runs_and_resumes(tmp_path):
    _skip_no_ephem()
    from kalachakra.local_autoencoder.training import load_checkpoint, train
    cfg = _tiny_cfg(tmp_path)
    final = train(cfg, max_steps=3)
    assert final.exists()
    _model_r, payload, _cfg = load_checkpoint(final, map_location="cpu")
    assert payload["step"] == 3 and np.isfinite(payload["metrics"]["loss"])
    # resume continues from step 3
    final2 = train(cfg, resume=str(final), max_steps=5)
    p2 = load_checkpoint(final2, map_location="cpu")[1]
    assert p2["step"] == 5


def test_inference_returns_color_and_attribution(tmp_path):
    _skip_no_ephem()
    from kalachakra.local_autoencoder.inference import LocalSkyInference
    from kalachakra.local_autoencoder.training import save_checkpoint
    cfg = _tiny_cfg(tmp_path)
    model = build_model(cfg.model)
    opt = torch.optim.AdamW(model.parameters())
    from kalachakra.local_autoencoder.training import cosine_warmup
    sch = cosine_warmup(opt, 1, 3)
    path = save_checkpoint(tmp_path / "ck.pt", model, opt, sch, 1, cfg)

    eng = LocalSkyInference.from_checkpoint(path, device="cpu")
    r = eng.infer("2024-04-08T18:17:00Z", 51.5, -0.12)
    assert len(r["oklab_color"]) == 3 and len(r["srgb8_color"]) == 3
    assert 0.0 <= r["oklab_color"][0] <= 1.0                # L in [0,1]
    assert abs(sum(r["attention"].values()) - 1.0) < 1e-4
    assert set(r["attention"]) == set(F.BODY_NAMES)
    assert np.asarray(r["attention_raw"]).shape == (2, 8, 11, 11)
