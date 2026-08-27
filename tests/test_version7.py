"""version7: the regional / city-grid engine.

Covers the curated city dataset, the structured node sampler (cities + regional lattice +
uniform fill), the reuse of version6's bounded/soft-clamped SIREN, and a tiny end-to-end
train -> checkpoint -> export (weights.json + cities.json + manifest.json) cycle that feeds
the texture-mapping frontend.
"""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import version7  # noqa: F401,E402  (installs the src path shim)
from version6.ephemeris import STATE_DIM                       # noqa: E402
from version7.cities import CITIES, unique_cities             # noqa: E402
from version7.config import (                                 # noqa: E402
    DataConfig, GridConfig, SirenConfig, TrainConfig, V7Config,
)
from version7.dataset import city_nodes, regional_grid        # noqa: E402


# ---------------------------------------------------------------------------
# curated cities
# ---------------------------------------------------------------------------
def test_cities_are_valid_and_deduped():
    assert len(CITIES) >= 100
    cs = unique_cities()
    names = [c[0] for c in cs]
    assert len(names) == len(set(names))                       # no duplicate names
    for _n, lat, lon, _r in cs:
        assert -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0
    regions = {c[3] for c in cs}
    assert {"Asia", "Europe", "Africa", "North America", "South America", "Oceania"} <= regions


# ---------------------------------------------------------------------------
# config (reuses version6 SirenConfig with its bounded L*a*b* head)
# ---------------------------------------------------------------------------
def test_config_defaults_and_roundtrip():
    cfg = V7Config()
    assert cfg.siren.in_features == STATE_DIM and cfg.siren.out_features == 3
    assert cfg.siren.lab_center == 50.0 and cfg.siren.lab_ab == 90.0    # gamut-bounded head
    assert cfg.grid.width == 180 and cfg.grid.height == 90
    assert 0.0 < cfg.data.city_frac < 1.0 and 0.0 < cfg.data.grid_frac < 1.0
    assert cfg.data.city_frac + cfg.data.grid_frac < 1.0               # leaves a uniform remainder
    d = cfg.to_dict()
    cfg2 = V7Config.from_dict(d)
    assert cfg2.grid.width == 180 and cfg2.siren.omega0 == cfg.siren.omega0


def test_from_dict_ignores_unknown_keys():
    cfg = V7Config.from_dict({"siren": {"hidden": 24, "bogus": 1}, "grid": {"width": 64, "x": 2}})
    assert cfg.siren.hidden == 24 and cfg.grid.width == 64


# ---------------------------------------------------------------------------
# structured node sampler
# ---------------------------------------------------------------------------
def test_regional_grid_covers_globe():
    g = regional_grid(30.0)
    assert g.shape[1] == 2
    assert g[:, 0].min() > -90 and g[:, 0].max() < 90
    assert g[:, 1].min() >= -180 and g[:, 1].max() <= 180
    assert len(g) == 6 * 12                                    # 6 lat bands x 12 lon columns


def test_city_nodes_match_cities():
    assert len(city_nodes()) == len(unique_cities())


def test_dataset_yields_structured_batches():
    from version7.dataset import build_dataloader
    cfg = DataConfig(batch=64, grid_step_deg=10.0, seed=1)
    loader = build_dataloader(cfg, num_workers=0)
    it = iter(loader)
    (a,) = next(it)
    (b,) = next(it)
    assert a.shape == (64, 33) and a.dtype == torch.float32 and torch.isfinite(a).all()
    assert not torch.equal(a, b)                               # fresh batch each step
    # each body triple is a unit vector (topocentric tensor sanity)
    v = a.numpy().reshape(64, 11, 3)
    assert np.allclose(np.linalg.norm(v, axis=-1), 1.0, atol=1e-4)


# ---------------------------------------------------------------------------
# end to end: train -> checkpoint -> export bundle
# ---------------------------------------------------------------------------
def _tiny_cfg(tmp_path, **train_kw):
    siren = SirenConfig(hidden=16, hidden_layers=2, omega0=30.0)
    grid = GridConfig(width=32, height=16)
    data = DataConfig(batch=48, grid_step_deg=15.0, seed=0)
    train = TrainConfig(max_steps=4, warmup_steps=1, save_every=100, log_every=1,
                        device="cpu", num_workers=0, out_dir=str(tmp_path / "run"), **train_kw)
    return V7Config(siren=siren, grid=grid, data=data, train=train)


def test_training_runs_and_resumes(tmp_path):
    from version7.training import CHECKPOINT_FORMAT, load_checkpoint, train
    cfg = _tiny_cfg(tmp_path)
    final = train(cfg, max_steps=4)
    assert final.exists()
    _m, payload, _c = load_checkpoint(final, map_location="cpu")
    assert payload["format"] == CHECKPOINT_FORMAT and payload["step"] == 4
    final2 = train(cfg, resume=str(final), max_steps=7)
    assert load_checkpoint(final2, map_location="cpu")[1]["step"] == 7


def test_model_output_is_gamut_bounded(tmp_path):
    from version7.training import load_checkpoint, train
    cfg = _tiny_cfg(tmp_path)
    final = train(cfg, max_steps=4)
    model, _p, _c = load_checkpoint(final, map_location="cpu")
    model.eval()
    with torch.no_grad():
        lab = model(torch.randn(512, 33) * 4.0)
    assert lab[:, 0].min() > 0.0 and lab[:, 0].max() < 100.0   # L* strictly in (0,100)
    assert lab[:, 1:].abs().max() < 90.0                       # a*,b* bounded -> no neon clipping


def test_export_bundle_for_frontend(tmp_path):
    from version7.training import export_manifest, export_weights_json, train
    cfg = _tiny_cfg(tmp_path)
    final = train(cfg, max_steps=4)
    web = tmp_path / "web"
    export_weights_json(str(final), str(web / "weights.json"))
    manifest = export_manifest(cfg, str(web))

    weights = json.loads((web / "weights.json").read_text())
    assert weights["output_activation"] == "lab_tanh" and "color_scale" in weights
    assert [ly["activation"] for ly in weights["layers"]] == ["sin", "sin", "linear"]

    cities = json.loads((web / "cities.json").read_text())
    assert len(cities) == len(unique_cities())
    assert {"name", "lat", "lon", "region"} <= set(cities[0])

    m = json.loads((web / "manifest.json").read_text())
    assert m == manifest
    assert m["grid"] == {"width": 32, "height": 16}
    assert m["n_bodies"] == 11 and len(m["bodies"]) == 11
    assert m["timeline"]["jd_start"] < m["timeline"]["jd_end"]
