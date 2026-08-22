"""
Training loop and checkpointing (blueprint §5.2, §5.3).

Drives the Spherical Autoencoder over the ephemeris stream in BF16 mixed
precision on the best available device (MPS on Apple Silicon, else CUDA, else
CPU). Two checkpoint tiers are written: lightweight *micro* checkpoints on a
wall-clock cadence to survive interruptions, and comprehensive *era* snapshots
every N simulated years for the parallel evaluation daemon (§5.3) to pick up.

Requires PyTorch.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..losses.geometric import CompositeGeodesicLoss, LossWeights
from ..models.autoencoder import SphericalAutoencoder
from .optim import OptimConfig, build_optimizer, build_scheduler


def select_device() -> torch.device:
    """Prefer Apple MPS (the target hardware), then CUDA, then CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class TrainConfig:
    optim: OptimConfig = field(default_factory=OptimConfig)
    loss_weights: LossWeights = field(default_factory=LossWeights)
    amp_dtype: str = "bfloat16"                    # native mixed precision
    micro_checkpoint_seconds: float = 12 * 3600    # every 12 wall-clock hours
    era_checkpoint_years: int = 500                # every 500 simulated years
    grad_clip: float = 1.0
    log_every: int = 50


class Trainer:
    """Coordinates model, optimizer, scheduler, loss and checkpointing."""

    def __init__(self, model: SphericalAutoencoder, cfg: TrainConfig,
                 checkpoint_dir: str | Path, device: torch.device | None = None):
        self.cfg = cfg
        self.device = device or select_device()
        self.model = model.to(self.device)
        self.optimizer = build_optimizer(self.model.parameters(), cfg.optim)
        self.scheduler = build_scheduler(self.optimizer, cfg.optim)
        self.criterion = CompositeGeodesicLoss(cfg.loss_weights)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.step = 0
        self._last_micro = time.monotonic()
        self._amp_dtype = getattr(torch, cfg.amp_dtype)

    # -- checkpoints ------------------------------------------------------
    def _payload(self) -> dict:
        return {
            "step": self.step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }

    def save_micro(self) -> Path:
        path = self.checkpoint_dir / f"micro_{self.step:012d}.pt"
        torch.save(self._payload(), path)
        # Keep a stable "latest" pointer for resume.
        torch.save(self._payload(), self.checkpoint_dir / "micro_latest.pt")
        return path

    def save_era(self, sim_year: int) -> Path:
        path = self.checkpoint_dir / f"era_{sim_year:06d}yr.pt"
        torch.save(self._payload(), path)
        return path

    def maybe_micro_checkpoint(self) -> None:
        now = time.monotonic()
        if now - self._last_micro >= self.cfg.micro_checkpoint_seconds:
            self.save_micro()
            self._last_micro = now

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.step = ckpt.get("step", 0)

    # -- optimization -----------------------------------------------------
    def train_step(self, e: torch.Tensor, lons: torch.Tensor) -> dict[str, float]:
        self.model.train()
        e = e.to(self.device, non_blocking=True)
        lons = lons.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=self.device.type, dtype=self._amp_dtype):
            recon, _z = self.model(e)
            # Reshape channel axis back to (..., B, 5) for the geometric loss.
            recon_f = recon.unflatten(-1, (-1, 5))
            target_f = e.unflatten(-1, (-1, 5))
            total, parts = self.criterion(recon_f, target_f)

        total.backward()
        if self.cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.optimizer.step()
        self.scheduler.step(self.step)
        self.step += 1
        self.maybe_micro_checkpoint()
        return {k: float(v.detach()) for k, v in parts.items()}

    def fit(self, loader: DataLoader, max_steps: int | None = None):
        """Iterate the stream, returning the last loss dict (for smoke tests)."""
        last: dict[str, float] = {}
        for e, lons in loader:
            last = self.train_step(e, lons)
            if self.step % self.cfg.log_every == 0:
                lr = self.scheduler.get_last_lr()[0]
                print(f"step={self.step} lr={lr:.2e} loss={last.get('total'):.5f}")
            if max_steps is not None and self.step >= max_steps:
                break
        return last
