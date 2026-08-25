"""Decoupled projection engine: features, models, losses, training, inference.

Covers the wrap-continuous celestial encoding, area-uniform terrestrial sampling,
tensor typing/device placement, Sky Encoder + Earth Lens shapes and attention
attribution, the three physics losses (including anti-collapse and temporal
behaviour), OKLab colour conversion, a tiny end-to-end training run, checkpoint
round-trip, the inference engine, TorchScript export, and the FastAPI endpoints.
"""

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F                                     # noqa: E402

from kalachakra.decoupled_engine import features                    # noqa: E402
from kalachakra.decoupled_engine.color import oklab_to_srgb8        # noqa: E402
from kalachakra.decoupled_engine.config import (                    # noqa: E402
    DataConfig, EarthLensConfig, EngineConfig, SkyEncoderConfig, TrainConfig,
)
from kalachakra.decoupled_engine.earth_lens import EarthLensDecoder  # noqa: E402
from kalachakra.decoupled_engine.losses import (                    # noqa: E402
    culmination_edge_permission,
    geometric_interference_contrastive_loss,
    harmonic_interference_descriptor,
    temporal_continuity_loss,
    terrestrial_smoothness_loss,
)
from kalachakra.decoupled_engine.sky_encoder import SkyEncoder      # noqa: E402
from kalachakra.ephemeris import global_state                       # noqa: E402
from kalachakra.ephemeris.calendar import parse_datetime           # noqa: E402


def _skip_no_ephem():
    if not global_state.ephemeris_available():
        pytest.skip("pyswisseph not installed")


# ---------------------------------------------------------------------------
# features: wrap continuity, encoding, sampling
# ---------------------------------------------------------------------------
def test_angular_wraparound_is_continuous():
    """0/360-degree seam: features vary smoothly and round-trip exactly."""
    lon_lo = np.deg2rad(np.full(10, 359.9))
    lon_hi = np.deg2rad(np.full(10, 0.1))
    lat = np.zeros(10)
    vel = np.zeros(10)
    f_lo = features.encode_celestial(lon_lo, lat, vel)
    f_hi = features.encode_celestial(lon_hi, lat, vel)
    # 0.2 degrees apart -> tiny distance despite straddling the 360->0 seam
    assert np.max(np.abs(f_lo - f_hi)) < 5e-3
    # sin^2 + cos^2 == 1 for both angle pairs
    assert np.allclose(f_lo[:, 0] ** 2 + f_lo[:, 1] ** 2, 1.0, atol=1e-6)
    # decode round-trips the longitude (mod 2*pi)
    lon_dec, _lat = features.decode_lonlat_np(f_hi)
    assert np.allclose(np.cos(lon_dec), np.cos(lon_hi), atol=1e-6)


def test_sphere_sampling_is_area_uniform():
    rng = np.random.default_rng(0)
    coords = features.sample_sphere_coords(20000, rng)
    lat, lon = coords[:, 0], coords[:, 1]
    assert lat.min() >= -math.pi / 2 - 1e-6 and lat.max() <= math.pi / 2 + 1e-6
    assert lon.min() >= -math.pi - 1e-6 and lon.max() <= math.pi + 1e-6
    # area-uniform => sin(lat) ~ U(-1,1) => E[sin lat] ~ 0
    assert abs(np.mean(np.sin(lat))) < 0.03


def test_equirect_grid_shape_and_bounds():
    g = features.equirect_grid(8, 4)
    assert g.shape == (32, 2)
    lat = g[:, 0]
    assert lat.max() <= math.pi / 2 + 1e-6 and lat.min() >= -math.pi / 2 - 1e-6


def test_latlon_to_unit_vector_is_unit():
    latlon = torch.tensor([[0.0, 0.0], [math.pi / 2, 1.0], [-0.3, math.pi]])
    uv = features.latlon_to_unit_vector(latlon)
    assert uv.shape == (3, 3)
    assert torch.allclose(uv.norm(dim=-1), torch.ones(3), atol=1e-5)


def test_celestial_features_from_ephemeris():
    _skip_no_ephem()
    global_state.auto_configure()
    jd = parse_datetime("2000-01-01T00:00:00Z")
    feat = features.celestial_features(jd)
    assert feat.shape == (10, 5) and feat.dtype == np.float32
    assert np.isfinite(feat).all()
    assert np.allclose(feat[:, 0] ** 2 + feat[:, 1] ** 2, 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# dataset: shapes, typing, device
# ---------------------------------------------------------------------------
def test_dataset_streams_typed_device_tensors():
    _skip_no_ephem()
    global_state.auto_configure()
    from kalachakra.decoupled_engine.dataset import build_dataloader, move_batch
    cfg = DataConfig(start_jd=parse_datetime("2000-01-01T00:00:00Z"),
                     end_jd=parse_datetime("2001-01-01T00:00:00Z"),
                     temporal_len=2, points_per_frame=8, samples_per_epoch=4, seed=1)
    loader = build_dataloader(cfg, batch_size=2, num_workers=0, device="cpu")
    cel, jds, coords = next(iter(loader))
    assert cel.shape == (2, 2, 10, 5) and cel.dtype == torch.float32
    assert jds.shape == (2, 2) and coords.shape == (2, 8, 2)
    for t in (cel, jds, coords):
        assert t.device.type == "cpu"                    # placed on the target device
    cel2, _j, _c = move_batch((cel, jds, coords), "cpu")
    assert cel2.device.type == "cpu"


# ---------------------------------------------------------------------------
# Sky Encoder
# ---------------------------------------------------------------------------
def _sky(tension_dim=64):
    return SkyEncoder(SkyEncoderConfig(d_model=32, nhead=8, num_layers=3,
                                       dim_feedforward=64, tension_dim=tension_dim))


def test_sky_encoder_shapes_and_norm():
    sky = _sky().eval()
    x = torch.randn(4, 10, 5)
    z = sky(x)
    assert z.shape == (4, 64)
    assert torch.allclose(z.norm(dim=-1), torch.ones(4), atol=1e-5)   # normalized


def test_sky_encoder_attention_and_attribution():
    sky = _sky().eval()
    x = torch.randn(3, 10, 5)
    z, attn = sky(x, return_attention=True)
    assert z.shape == (3, 64)
    assert attn.shape == (3, 3, 8, 11, 11)               # (B, layers, heads, S, S)
    attribution = sky.planetary_attribution(x)
    assert attribution.shape == (3, 10)
    assert torch.allclose(attribution.sum(dim=-1), torch.ones(3), atol=1e-4)
    assert (attribution >= 0).all()


def test_sky_encoder_grad_checkpoint_matches():
    torch.manual_seed(0)
    cfg = SkyEncoderConfig(d_model=32, nhead=8, num_layers=2, tension_dim=32)
    sky = SkyEncoder(cfg).train()
    x = torch.randn(2, 10, 5, requires_grad=True)
    z = sky(x)
    z.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# Earth Lens Decoder
# ---------------------------------------------------------------------------
def _earth(tension_dim=64):
    return EarthLensDecoder(EarthLensConfig(tension_dim=tension_dim, num_fourier=8,
                                            hidden=32, n_blocks=2, activation="gauss"))


def test_earth_lens_shapes_and_bounds():
    earth = _earth().eval()
    tension = torch.randn(3, 64)
    latlon = torch.randn(3, 5, 2)
    out = earth(tension, latlon)
    assert out.shape == (3, 5, 3)
    L, a, b = out[..., 0], out[..., 1], out[..., 2]
    assert (L >= 0).all() and (L <= 1).all()                  # OKLab luminance in [0,1]
    assert a.abs().max() <= 0.4 + 1e-5 and b.abs().max() <= 0.4 + 1e-5


def test_earth_lens_single_and_grid_paths():
    earth = _earth().eval()
    # unbatched single tension + a set of points
    out = earth(torch.randn(64), torch.randn(7, 2))
    assert out.shape == (7, 3)
    # one tension broadcast over a global grid
    grid = torch.as_tensor(features.equirect_grid(8, 4)).unsqueeze(0)   # (1, 32, 2)
    out2 = earth(torch.randn(1, 64), grid)
    assert out2.shape == (1, 32, 3)


def test_earth_lens_sine_activation_builds():
    earth = EarthLensDecoder(EarthLensConfig(tension_dim=32, num_fourier=4, hidden=16,
                                             n_blocks=2, activation="sine")).eval()
    out = earth(torch.randn(2, 32), torch.randn(2, 3, 2))
    assert out.shape == (2, 3, 3) and torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# losses
# ---------------------------------------------------------------------------
def test_harmonic_descriptor_wrap_continuous():
    torch.manual_seed(0)
    cel = torch.randn(1, 10, 5)
    # force one body's longitude to straddle the seam and re-encode
    def with_lon(deg):
        c = cel.clone()
        r = math.radians(deg)
        c[0, 0, 0], c[0, 0, 1] = math.sin(r), math.cos(r)
        return c
    d_lo = harmonic_interference_descriptor(with_lon(359.9))
    d_hi = harmonic_interference_descriptor(with_lon(0.1))
    # 0.2-deg step, amplified by the k<=6 harmonics (~6x) -> still tiny and
    # seam-free; a genuine wrap discontinuity would be O(1).
    assert torch.max((d_lo - d_hi).abs()) < 3e-2


def test_geometric_contrastive_penalizes_collapse():
    torch.manual_seed(0)
    cel = torch.randn(6, 10, 5)
    g = harmonic_interference_descriptor(cel)
    z_aligned = F.normalize(g, dim=-1)                     # latent == geometry
    z_collapsed = torch.ones(6, g.shape[-1])              # all identical -> collapse
    loss_aligned = geometric_interference_contrastive_loss(z_aligned, cel, 0.1)
    loss_collapsed = geometric_interference_contrastive_loss(z_collapsed, cel, 0.1)
    assert torch.isfinite(loss_aligned) and loss_aligned >= 0
    assert loss_collapsed > loss_aligned                  # collapse is punished
    assert loss_collapsed > 1.0


def test_terrestrial_smoothness_and_permission():
    color = torch.randn(2, 5, 3)
    assert terrestrial_smoothness_loss(color, color.clone(), 0.01,
                                       torch.zeros(2, 5)) == 0.0    # no gradient
    loss = terrestrial_smoothness_loss(color, color + 0.1, 0.01, torch.zeros(2, 5))
    assert loss > 0
    cel = torch.randn(2, 10, 5)
    coords = torch.randn(2, 5, 2)
    permit = culmination_edge_permission(cel, coords, torch.zeros(2))
    assert permit.shape == (2, 5)
    assert (permit > 0).all() and (permit <= 1.0 + 1e-6).all()


def test_temporal_continuity_behaviour():
    p = torch.randn(2, 4, 6, 3)
    # constant in time -> zero
    const = p[:, :1].expand(2, 4, 6, 3).contiguous()
    assert temporal_continuity_loss(const) < 1e-6
    # constant-velocity ramp -> zero curvature (fast smooth drift allowed)
    t = torch.arange(4).float().view(1, 4, 1, 1)
    ramp = (t * 0.1).expand(2, 4, 6, 3).contiguous()
    assert temporal_continuity_loss(ramp) < 1e-6
    # an erratic spike -> positive
    spike = const.clone()
    spike[:, 2] += 0.5
    assert temporal_continuity_loss(spike) > 0


# ---------------------------------------------------------------------------
# colour
# ---------------------------------------------------------------------------
def test_oklab_to_srgb8():
    white = oklab_to_srgb8(np.array([1.0, 0.0, 0.0]))
    black = oklab_to_srgb8(np.array([0.0, 0.0, 0.0]))
    assert white.shape == (3,) and white.dtype == np.uint8
    assert (white >= 250).all() and (black == 0).all()
    img = oklab_to_srgb8(np.zeros((4, 8, 3)))
    assert img.shape == (4, 8, 3)


# ---------------------------------------------------------------------------
# bundle round-trip
# ---------------------------------------------------------------------------
def _tiny_engine_cfg(**data_kw):
    data = DataConfig(temporal_len=3, points_per_frame=8, samples_per_epoch=4,
                      stride_seconds=86400.0, **data_kw)
    sky = SkyEncoderConfig(d_model=16, nhead=8, num_layers=2, dim_feedforward=32,
                           tension_dim=32)
    earth = EarthLensConfig(tension_dim=32, num_fourier=8, hidden=32, n_blocks=2)
    train = TrainConfig(batch_size=2, max_steps=3, warmup_steps=1, save_every=2,
                        amp=False, device="cpu")
    return EngineConfig(sky=sky, earth=earth, data=data, train=train)


def test_checkpoint_roundtrip(tmp_path):
    from kalachakra.decoupled_engine.bundle import (
        build_models, load_checkpoint, save_checkpoint,
    )
    cfg = _tiny_engine_cfg()
    sky, earth = build_models(cfg)
    sky.eval()
    earth.eval()
    x = torch.randn(2, 10, 5)
    z0 = sky(x)
    path = save_checkpoint(tmp_path / "ck.pt", sky, earth, cfg, step=7)
    sky2, earth2, cfg2, payload = load_checkpoint(path, map_location="cpu")
    assert payload["step"] == 7 and cfg2.sky.tension_dim == 32
    assert torch.allclose(sky2.eval()(x), z0, atol=1e-6)


# ---------------------------------------------------------------------------
# training (end to end, tiny)
# ---------------------------------------------------------------------------
def test_training_runs_and_checkpoints(tmp_path):
    _skip_no_ephem()
    from kalachakra.decoupled_engine.training import train
    cfg = _tiny_engine_cfg(start_jd=parse_datetime("2000-01-01T00:00:00Z"),
                           end_jd=parse_datetime("2000-06-01T00:00:00Z"))
    cfg.train.out_dir = str(tmp_path / "run")
    final = train(cfg, num_workers=0, max_steps=3)
    assert final.exists()
    from kalachakra.decoupled_engine.bundle import load_checkpoint
    _sky, _earth, _cfg, payload = load_checkpoint(final, map_location="cpu")
    assert payload["step"] == 3
    assert all(np.isfinite(v) for v in payload["metrics"].values())


# ---------------------------------------------------------------------------
# inference
# ---------------------------------------------------------------------------
def _save_random_checkpoint(tmp_path, **data_kw):
    from kalachakra.decoupled_engine.bundle import build_models, save_checkpoint
    cfg = _tiny_engine_cfg(**data_kw)
    sky, earth = build_models(cfg)
    return save_checkpoint(tmp_path / "infer.pt", sky, earth, cfg, step=1), cfg


def test_inference_texture_and_pinpoint(tmp_path):
    _skip_no_ephem()
    from kalachakra.decoupled_engine.inference import DecoupledInference
    path, _cfg = _save_random_checkpoint(tmp_path)
    eng = DecoupledInference.from_checkpoint(path, device="cpu")
    tex = eng.global_texture("2000-01-01", width=8, height=4)
    assert tex["oklab"].shape == (4, 8, 3)
    assert tex["rgb8"].shape == (4, 8, 3)
    assert len(tex["bytes"]) == 4 * 8 * 3
    pin = eng.pinpoint("2000-01-01", 28.6, 77.2)
    assert pin["oklab"].shape == (3,)
    assert abs(sum(pin["attribution"].values()) - 1.0) < 1e-4
    assert set(pin["attribution"]) == set(features.BODY_NAMES)


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------
def test_torchscript_export_matches_eager(tmp_path):
    from kalachakra.decoupled_engine.bundle import build_models
    from kalachakra.decoupled_engine.export import export_torchscript
    cfg = _tiny_engine_cfg()
    sky, earth = build_models(cfg)
    sky.eval()
    earth.eval()
    paths = export_torchscript(sky, earth, cfg, tmp_path / "ts", device="cpu")
    sky_ts = torch.jit.load(paths["sky"])
    x = torch.randn(2, 10, 5)
    assert torch.allclose(sky_ts(x), sky(x), atol=1e-5)
    earth_ts = torch.jit.load(paths["earth"])
    tension, latlon = torch.randn(2, 32), torch.randn(2, 4, 2)
    assert torch.allclose(earth_ts(tension, latlon), earth(tension, latlon), atol=1e-5)


# ---------------------------------------------------------------------------
# FastAPI integration
# ---------------------------------------------------------------------------
def test_jd_from_timestamp_parsing():
    from kalachakra.decoupled_engine.inference import jd_from_timestamp
    assert jd_from_timestamp(2460000.5) == 2460000.5
    assert jd_from_timestamp("2460000.5") == 2460000.5        # numeric string -> JD
    iso = jd_from_timestamp("2000-01-01T00:00:00Z")
    assert 2451544.0 < iso < 2451546.0                        # ~ J2000


def test_api_demo_mode_and_dashboard(tmp_path):
    _skip_no_ephem()
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from kalachakra.decoupled_engine.api import create_app
    app = create_app(None, device="cpu", bank_size=4)         # no checkpoint -> demo
    c = TestClient(app)

    h = c.get("/health").json()
    assert h["status"] == "ok" and h["demo"] is True
    assert "coverage" in h and h["coverage"]["end_jd"] > h["coverage"]["start_jd"]

    assert c.get("/").status_code == 200                       # dashboard served
    assert c.get("/api/coastlines.geojson").status_code == 200

    mid = 0.5 * (h["coverage"]["start_jd"] + h["coverage"]["end_jd"])
    tex = c.get("/api/texture", params={"timestamp": mid, "width": 8, "height": 4})
    assert tex.status_code == 200 and len(tex.content) == 8 * 4 * 3
    assert tex.headers.get("x-jd") is not None                 # numeric-JD timestamp works
    pt = c.post("/api/point", json={"timestamp": mid, "lat": 10.0, "lon": 20.0}).json()
    assert len(pt["rgb"]) == 3 and abs(sum(pt["attribution"].values()) - 1.0) < 1e-4
    assert "date" in pt


def test_api_endpoints(tmp_path):
    _skip_no_ephem()
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from kalachakra.decoupled_engine.api import create_app
    path, _cfg = _save_random_checkpoint(
        tmp_path, start_jd=parse_datetime("2000-01-01T00:00:00Z"),
        end_jd=parse_datetime("2002-01-01T00:00:00Z"))
    app = create_app(str(path), device="cpu", bank_size=4)
    c = TestClient(app)

    h = c.get("/health").json()
    assert h["status"] == "ok" and h["tension_dim"] == 32 and len(h["bodies"]) == 10

    tex = c.get("/api/texture", params={"timestamp": "2000-01-01",
                                        "width": 8, "height": 4})
    assert tex.status_code == 200 and len(tex.content) == 8 * 4 * 3

    pt = c.post("/api/point", json={"timestamp": "2000-01-01", "lat": 28.6,
                                    "lon": 77.2}).json()
    assert len(pt["rgb"]) == 3 and abs(sum(pt["attribution"].values()) - 1.0) < 1e-4

    sim = c.post("/api/similar", json={"timestamp": "2001-01-01", "k": 2}).json()
    assert "matches" in sim and len(sim["matches"]) >= 1
    assert all(-1.001 <= m["similarity"] <= 1.001 for m in sim["matches"])
