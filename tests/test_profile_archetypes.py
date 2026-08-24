"""Archetype profiler: tiny v3 checkpoint -> dossiers dict, schema + math checks.

Exercises scripts/profile_archetypes.build_dossiers end-to-end on a small mesh:
top-5 token selection, empirical (pre-quantization) magnitude, latitudinal
affinity, and the planetary-driver feature->body mapping.
"""

import importlib.util
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from kalachakra.ephemeris import global_state                        # noqa: E402
from kalachakra.ephemeris.bodies import ENTITIES                     # noqa: E402
from kalachakra.grid import geodesic                                 # noqa: E402
from kalachakra.models.autoencoder_v3 import (                       # noqa: E402
    VQAutoencoderV3, VQAutoencoderV3Config, build_knn,
)


def _load_profiler():
    p = Path(__file__).resolve().parents[1] / "scripts" / "profile_archetypes.py"
    spec = importlib.util.spec_from_file_location("profile_archetypes", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tiny_checkpoint(path, n=600):
    grid = geodesic.fibonacci_sphere(n)
    nb = build_knn(grid.xyz, 7)
    cfg = VQAutoencoderV3Config(n_nodes=n, in_features=50, hidden=16, latent=64,
                                fourier_modes=6, knn=7, n_blocks=2, codebook_size=4096,
                                node_chunk=128, vq_chunk=4096)
    model = VQAutoencoderV3(cfg, nb)
    torch.save({
        "format": "kalachakra-vqmodel-v3",
        "projection_version": 2,
        "config": asdict(cfg),
        "neighbors": np.asarray(nb, dtype=np.int64),
        "grid_xyz": np.asarray(grid.xyz, dtype=np.float64),
        "state_dict": model.state_dict(),
        "step": 25,
    }, path)
    return n


def test_profile_archetypes_end_to_end(tmp_path):
    if not global_state.ephemeris_available():
        pytest.skip("pyswisseph not installed")
    prof = _load_profiler()
    ckpt = tmp_path / "model_step_000025.pt"
    n = _tiny_checkpoint(ckpt)

    dossiers, meta = prof.build_dossiers(
        str(ckpt), n_days=12, n_years=500, batch=6, seed=0,
        device=torch.device("cpu"))

    assert 1 <= len(dossiers) <= 5
    assert meta["n_nodes"] == n
    assert meta["codebook_unit_normalized"] is True
    assert meta["magnitude_source"] == "mean_pre_quantization_latent_norm"

    total_cover = 0.0
    names = {e.name for e in ENTITIES}
    mags = []
    for tid, d in dossiers.items():
        assert d["id"] == int(tid)
        ph = d["physics"]
        # empirical magnitude is a real pre-quant latent norm, not the unit codebook
        assert ph["magnitude"] > 1.0
        assert 0.0 <= ph["magnitude_percentile"] <= 100.0
        assert ph["variance"] > 0.0
        mags.append(ph["magnitude"])

        sp = d["spatial"]
        assert 0.0 < sp["current_coverage_percent"] <= 100.0
        assert -90.0 <= sp["mean_latitude_deg"] <= 90.0
        assert sp["std_latitude_deg"] >= 0.0
        total_cover += sp["current_coverage_percent"]

        # "<Body> (Feature Index <f>)" and f // 5 must name that same body
        driver = d["planetary_driver"]
        body, rest = driver.split(" (Feature Index ")
        fidx = int(rest.rstrip(")"))
        assert body in names
        assert 0 <= fidx < 50
        assert ENTITIES[fidx // 5].name == body
        assert -1.0 <= d["planetary_driver_correlation"] <= 1.0

    # the top-5 cannot cover more than the whole globe
    assert total_cover <= 100.0 + 1e-6
    # empirical magnitudes actually differ across tokens (not the flat unit norm)
    assert max(mags) - min(mags) > 1e-6


def test_profiler_feature_to_body_indexing():
    # feature f belongs to body f // 5 across all 50 features / 10 bodies
    assert ENTITIES[0].name == "Sun" and ENTITIES[5].name == "Jupiter"
    assert ENTITIES[24 // 5].name == "Mars"
    assert ENTITIES[32 // 5].name == "Saturn"
    assert ENTITIES[49 // 5].name == "Ayanamsha"
