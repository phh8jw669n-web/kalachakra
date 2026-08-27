"""version6: the continuous SIREN engine, end to end.

Covers the self-contained analytic topocentric ephemeris (shape/units, batch
consistency, GMST, and an altitude cross-check against pyswisseph near the present),
the SIREN network (shapes + the export→re-run parity that guarantees the JS/GLSL ports
match PyTorch), the isometric loss + gauge anchor, the stochastic generator, and a tiny
end-to-end train → checkpoint → weight/golden export cycle.
"""

import json
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import version6  # noqa: F401,E402  (installs the src path shim)
from version6 import ephemeris as ephem                       # noqa: E402
from version6.config import (                                 # noqa: E402
    DataConfig, SirenConfig, TrainConfig, V6Config,
)
from version6.losses import anchor_loss, color_stats, isometric_loss   # noqa: E402
from version6.siren import Siren, build_siren                 # noqa: E402


# ---------------------------------------------------------------------------
# ephemeris: geometry, units, batch consistency
# ---------------------------------------------------------------------------
def test_state_geometry_constants():
    assert ephem.N_BODIES == 11 and ephem.STATE_DIM == 33
    assert len(ephem.BODY_NAMES) == 11
    c = SirenConfig()
    assert c.in_features == 33 and c.out_features == 3 and c.omega0 == 30.0


def test_topocentric_tensor_shape_and_unit_vectors():
    rng = np.random.default_rng(0)
    n = 128
    lat = rng.uniform(-90, 90, n)
    lon = rng.uniform(-180, 180, n)
    jd = rng.uniform(ephem.J2000 - 1e6, ephem.J2000 + 1e6, n)
    sky = ephem.topocentric_tensor(lat, lon, jd)
    assert sky.shape == (n, 33) and sky.dtype == np.float32
    assert np.isfinite(sky).all()
    # each body is a (North, East, Up) unit vector
    v = sky.reshape(n, 11, 3)
    norms = np.linalg.norm(v, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-4)
    assert np.abs(v[..., 2]).max() <= 1.0 + 1e-6            # Up == sin(altitude)


def test_topocentric_batch_matches_per_sample():
    # the JS/GLSL ports run one sample per call, so the vectorised batch MUST equal
    # the concatenation of single-sample evaluations.
    lat = np.array([48.8566, -33.87, 0.0, 78.2])
    lon = np.array([2.3522, 151.2, -0.0, 15.6])
    jd = np.array([2451545.0, 2460000.5, 2440000.0, 2500000.5])
    batch = ephem.topocentric_tensor(lat, lon, jd)
    for i in range(len(lat)):
        one = ephem.topocentric_tensor(lat[i:i + 1], lon[i:i + 1], jd[i:i + 1])
        assert np.allclose(batch[i], one[0], atol=1e-6)


def test_gmst_at_j2000():
    # GMST at J2000.0 is the well-known 280.4606° (Meeus).
    g = float(ephem.gmst_deg(np.array([ephem.J2000]))[0])
    assert abs((g - 280.46062 + 180) % 360 - 180) < 1e-3


def test_altitude_matches_swisseph_near_present():
    swe = pytest.importorskip("swisseph")
    flags = swe.FLG_SWIEPH | swe.FLG_TOPOCTR | swe.FLG_EQUATORIAL
    swe_ids = [swe.SUN, swe.MOON, swe.MERCURY, swe.VENUS, swe.MARS,
               swe.JUPITER, swe.SATURN, swe.URANUS, swe.NEPTUNE, swe.PLUTO]
    # deterministic observers/times within a few decades of J2000, where our closed-form
    # ephemeris is arc-minute class (it degrades smoothly over millennia by design).
    pts = [(48.8566, 2.3522, 2451545.0), (51.5074, -0.1278, 2456000.5),
           (-33.8688, 151.2093, 2459000.25), (35.6762, 139.6503, 2447000.0),
           (0.0, 0.0, 2455000.75), (40.7128, -74.0060, 2458849.5)]
    for lat, lon, jd in pts:
        swe.set_topo(lon, lat, 0.0)
        geopos = (lon, lat, 0.0)
        sky = ephem.topocentric_tensor(np.array([lat]), np.array([lon]), np.array([jd]))[0]
        for i, sid in enumerate(swe_ids):
            ra, dec, dist = swe.calc_ut(jd, sid, flags)[0][:3]
            _az, alt_true, _app = swe.azalt(jd, swe.EQU2HOR, geopos, 0.0, 0.0, (ra, dec, dist))
            alt_ours = math.degrees(math.asin(max(-1.0, min(1.0, float(sky[i * 3 + 2])))))
            tol = 1.6 if sid == swe.MOON else 0.6     # Moon uses a short low-precision series
            assert abs(alt_ours - alt_true) < tol, (ephem.BODY_NAMES[i], jd, alt_ours, alt_true)


# ---------------------------------------------------------------------------
# SIREN: shapes + export/re-run parity (the JS + GLSL contract)
# ---------------------------------------------------------------------------
def test_siren_shapes_and_export_structure():
    net = build_siren(in_features=33, hidden=24, hidden_layers=2, out_features=3, omega0=30.0)
    y = net(torch.randn(7, 33))
    assert y.shape == (7, 3)
    w = net.export_weights()
    assert w["omega0"] == 30.0 and w["in_features"] == 33 and w["out_features"] == 3
    # hidden_layers sin layers + 1 linear output layer
    assert len(w["layers"]) == 3
    assert [ly["activation"] for ly in w["layers"]] == ["sin", "sin", "linear"]
    assert len(w["layers"][0]["W"]) == 24 and len(w["layers"][0]["W"][0]) == 33
    assert len(w["layers"][-1]["W"]) == 3 and len(w["layers"][-1]["b"]) == 3


def _numpy_siren(weights, x):
    """A dependency-free re-run of the exported weights — mirrors makeSiren + boundLab in
    siren6.js and the shader (linear logits, then the bounded L*a*b* head)."""
    omega0 = weights["omega0"]
    h = np.asarray(x, dtype=np.float64)
    for layer in weights["layers"]:
        W = np.asarray(layer["W"], dtype=np.float64)      # [out][in]
        b = np.asarray(layer["b"], dtype=np.float64)
        h = h @ W.T + b
        if layer["activation"] == "sin":
            h = np.sin(omega0 * h)
    if weights.get("output_activation") == "lab_tanh":
        lc, ls, ab = weights["lab_center"], weights["lab_lspan"], weights["lab_ab"]
        h = np.stack([lc + ls * np.tanh(h[..., 0] / ls),
                      ab * np.tanh(h[..., 1] / ab),
                      ab * np.tanh(h[..., 2] / ab)], axis=-1)
    return h


def test_export_weights_match_torch_forward():
    torch.manual_seed(1)
    net = build_siren(hidden=32, hidden_layers=3).eval()
    w = net.export_weights()
    x = torch.randn(16, 33)
    with torch.no_grad():
        ref = net(x).numpy()
    got = _numpy_siren(w, x.numpy())
    # this parity is exactly what lets the browser/GLSL reproduce PyTorch bit-for-bit
    assert np.max(np.abs(ref - got)) < 1e-4


def test_bounded_lab_head_is_in_gamut_and_near_identity():
    from version6.siren import bound_lab
    # extreme logits must still land inside (0,100) x (-90,90)^2
    z = torch.randn(20000, 3) * 40.0
    lab = bound_lab(z, 50.0, 50.0, 90.0)
    assert lab[:, 0].min() > 0.0 and lab[:, 0].max() < 100.0
    assert lab[:, 1].abs().max() < 90.0 and lab[:, 2].abs().max() < 90.0
    # slope-1 (near-identity) around the centre so the metric is preserved for small colours
    small = torch.tensor([[2.0, 1.0, -1.5]])
    out = bound_lab(small, 50.0, 50.0, 90.0)
    assert abs(float(out[0, 0]) - 52.0) < 0.05     # L ~= 50 + z
    assert abs(float(out[0, 1]) - 1.0) < 0.02 and abs(float(out[0, 2]) + 1.5) < 0.02


def test_siren_forward_is_bounded():
    net = build_siren(hidden=16).eval()
    lab = net(torch.randn(256, 33) * 5.0)
    assert lab[:, 0].min() > 0.0 and lab[:, 0].max() < 100.0
    assert lab[:, 1:].abs().max() < 90.0


def test_siren_init_is_bounded():
    net = Siren(in_features=33, hidden=48, hidden_layers=2, omega0=30.0)
    layers = list(net.net)
    first = layers[0].lin.weight.detach().abs().max().item()
    assert first <= 1.0 / 33 + 1e-6                        # first layer U(-1/in, 1/in)
    hid_bound = math.sqrt(6.0 / 48) / 30.0
    assert layers[1].lin.weight.detach().abs().max().item() <= hid_bound + 1e-6


# ---------------------------------------------------------------------------
# isometric loss + gauge anchor
# ---------------------------------------------------------------------------
def test_isometric_loss_properties():
    torch.manual_seed(0)
    s = torch.randn(48, 3)
    scale = 20.0
    # a perfect scaled isometry: colour distances == scale * physical distances -> ~0
    assert isometric_loss(s, scale * s, scale) < 1e-6
    # a collapsed (constant) colour is heavily penalised
    assert isometric_loss(s, torch.full((48, 3), 0.5), scale) > 1.0
    # general case is finite and non-negative
    loss = isometric_loss(torch.randn(48, 33), torch.randn(48, 3), scale)
    assert torch.isfinite(loss) and float(loss) >= 0.0


def test_anchor_loss_and_color_stats():
    anchored = torch.tensor([[60.0, 0.0, 0.0]]).repeat(10, 1)
    assert float(anchor_loss(anchored)) < 1e-9
    assert float(anchor_loss(torch.zeros(10, 3))) > 1.0
    stats = color_stats(torch.randn(32, 3, requires_grad=True))
    assert set(stats) == {"mean_L", "std_L", "std_a", "std_b"}
    assert all(isinstance(v, float) for v in stats.values())    # detached to plain floats


# ---------------------------------------------------------------------------
# stochastic generator (never a static dataset / grid)
# ---------------------------------------------------------------------------
def test_dataset_yields_fresh_batches():
    from version6.dataset import build_dataloader
    cfg = DataConfig(batch=16, seed=0)
    loader = build_dataloader(cfg, num_workers=0)
    it = iter(loader)
    (a,) = next(it)
    (b,) = next(it)
    assert a.shape == (16, 33) and a.dtype == torch.float32 and torch.isfinite(a).all()
    assert not torch.equal(a, b)               # a fresh random batch every step


def test_cosine_warmup_hits_lr_min():
    from version6.training import cosine_warmup
    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1e-4)
    sch = cosine_warmup(opt, warmup_steps=100, max_steps=1000, base_lr=1e-4, lr_min=1e-6)
    lrs = []
    for _ in range(1000):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sch.step()
    assert abs(lrs[99] - 1e-4) < 1e-9 and abs(max(lrs) - 1e-4) < 1e-9
    assert abs(lrs[-1] - 1e-6) < 2e-8


# ---------------------------------------------------------------------------
# end to end: train -> checkpoint -> resume -> weight/golden export
# ---------------------------------------------------------------------------
def _tiny_cfg(tmp_path, **train_kw):
    siren = SirenConfig(hidden=16, hidden_layers=2, omega0=30.0)
    data = DataConfig(batch=32, seed=0)
    train = TrainConfig(max_steps=4, warmup_steps=1, save_every=100, log_every=1,
                        device="cpu", num_workers=0, out_dir=str(tmp_path / "run"), **train_kw)
    return V6Config(siren=siren, data=data, train=train)


def test_training_runs_and_resumes(tmp_path):
    from version6.training import load_checkpoint, train
    cfg = _tiny_cfg(tmp_path)
    final = train(cfg, max_steps=4)
    assert final.exists()
    _m, payload, _cfg = load_checkpoint(final, map_location="cpu")
    assert payload["step"] == 4 and payload["format"] == "kalachakra-version6-siren"
    final2 = train(cfg, resume=str(final), max_steps=7)
    assert load_checkpoint(final2, map_location="cpu")[1]["step"] == 7


def test_weight_export_and_golden(tmp_path):
    from version6.export_weights import export
    from version6.training import load_checkpoint, train
    cfg = _tiny_cfg(tmp_path)
    final = train(cfg, max_steps=4)
    out = tmp_path / "weights.json"
    export(str(final), str(out))
    assert out.exists() and (tmp_path / "golden.json").exists()

    weights = json.loads(out.read_text())
    assert weights["color_scale"] == cfg.train.color_scale
    assert weights["output_activation"] == "lab_tanh"
    assert weights["lab_center"] == 50.0 and weights["lab_ab"] == 90.0
    assert [ly["activation"] for ly in weights["layers"]] == ["sin", "sin", "linear"]

    # golden points: the browser re-runs its own ephemeris + SIREN and must reproduce these
    golden = json.loads((tmp_path / "golden.json").read_text())
    model, _payload, _cfg = load_checkpoint(final, map_location="cpu")
    model.eval()
    for pt in golden["points"]:
        assert len(pt["sky"]) == 33 and len(pt["lab"]) == 3
        # the stored colour is bounded/displayable
        assert 0.0 < pt["lab"][0] < 100.0 and abs(pt["lab"][1]) < 90.0 and abs(pt["lab"][2]) < 90.0
        # our ephemeris reproduces the stored 33-D state for that (lat,lon,jd)
        sky = ephem.topocentric_tensor(np.array([pt["lat"]]), np.array([pt["lon"]]),
                                       np.array([pt["jd"]]))[0]
        assert np.allclose(sky, np.array(pt["sky"]), atol=1e-4)
        # the exported weights, re-run in pure numpy (matmul+sin then the bounded head),
        # reproduce the network colour — this is the JS/GLSL parity contract
        got = _numpy_siren(weights, np.array(pt["sky"]))
        assert np.allclose(got, np.array(pt["lab"]), atol=1e-3)
