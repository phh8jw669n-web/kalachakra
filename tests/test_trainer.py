"""Smoke test for the training loop. Skipped when torch is absent."""

import pytest

torch = pytest.importorskip("torch")

import kalachakra.constants as C                                    # noqa: E402
from kalachakra.grid import geodesic                               # noqa: E402
from kalachakra.models.autoencoder import (                        # noqa: E402
    AutoencoderConfig, SphericalAutoencoder,
)
from kalachakra.models.spherical_conv import build_knn             # noqa: E402
from kalachakra.training.optim import OptimConfig                  # noqa: E402
from kalachakra.training.trainer import TrainConfig, Trainer       # noqa: E402


def _fake_loader(n_batches=3, b=2, t=8, n=48):
    grid = geodesic.fibonacci_sphere(n)
    neighbors = build_knn(grid, 5)
    cfg = AutoencoderConfig(n_nodes=n, hidden=16, latent=C.LATENT_DIM,
                            fourier_modes=4, knn=5, n_blocks=1)
    model = SphericalAutoencoder(cfg, neighbors)
    batches = [
        (torch.randn(b, t, n, C.LOCAL_FIELD_WIDTH), torch.randn(b, t, C.N_BODIES))
        for _ in range(n_batches)
    ]
    return model, batches


def test_train_step_runs_and_reports_loss(tmp_path):
    model, batches = _fake_loader()
    tcfg = TrainConfig(
        optim=OptimConfig(optimizer="lion", lr=1e-3, restart_period=100),
        # Keep the checkpoint cadence out of the way of a short test.
        micro_checkpoint_seconds=1e9,
    )
    trainer = Trainer(model, tcfg, checkpoint_dir=tmp_path,
                      device=torch.device("cpu"))
    last = trainer.fit(batches, max_steps=3)
    assert "total" in last and last["total"] == last["total"]  # not NaN
    assert trainer.step == 3


def test_checkpoint_save_and_load_roundtrip(tmp_path):
    model, batches = _fake_loader(n_batches=1)
    tcfg = TrainConfig(micro_checkpoint_seconds=1e9)
    trainer = Trainer(model, tcfg, checkpoint_dir=tmp_path,
                      device=torch.device("cpu"))
    trainer.fit(batches, max_steps=1)
    path = trainer.save_era(sim_year=500)
    assert path.exists()

    trainer2 = Trainer(model, tcfg, checkpoint_dir=tmp_path,
                       device=torch.device("cpu"))
    trainer2.load(path)
    assert trainer2.step == trainer.step
