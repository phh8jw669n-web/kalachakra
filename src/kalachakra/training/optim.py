"""
Optimization: Lion optimizer with AdamW fallback + cosine annealing (blueprint §5.2).

Lion (EvoLved Sign Momentum) is memory-frugal — it stores a single momentum
buffer rather than Adam's two moments — which matters under the 80 GB MPS tensor
budget. AdamW is available as a drop-in fallback. Both pair with a cosine
annealing schedule with warm restarts to escape local minima at conjunction
shifts.

Requires PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


class Lion(Optimizer):
    """Lion optimizer (Chen et al., 2023). Decoupled weight decay, sign updates."""

    def __init__(self, params, lr: float = 1e-4,
                 betas: tuple[float, float] = (0.9, 0.99),
                 weight_decay: float = 0.0):
        if lr <= 0.0:
            raise ValueError("lr must be positive")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError("betas must be in [0, 1)")
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(p)
                m = state["momentum"]

                # Decoupled weight decay.
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                # Update uses the interpolated momentum's sign; buffer uses beta2.
                update = m.mul(beta1).add_(grad, alpha=1.0 - beta1).sign_()
                p.add_(update, alpha=-lr)
                m.mul_(beta2).add_(grad, alpha=1.0 - beta2)

        return loss


@dataclass
class OptimConfig:
    optimizer: str = "lion"          # "lion" | "adamw"
    lr: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.99)
    # Cosine annealing warm restarts.
    restart_period: int = 10_000     # steps until first restart (T_0)
    restart_mult: int = 2            # period multiplier per restart (T_mult)
    min_lr: float = 1e-6


def build_optimizer(params, cfg: OptimConfig) -> Optimizer:
    if cfg.optimizer.lower() == "lion":
        return Lion(params, lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay)
    if cfg.optimizer.lower() == "adamw":
        return AdamW(params, lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay)
    raise ValueError(f"unknown optimizer: {cfg.optimizer!r}")


def build_scheduler(optimizer: Optimizer, cfg: OptimConfig):
    return CosineAnnealingWarmRestarts(
        optimizer,
        T_0=cfg.restart_period,
        T_mult=cfg.restart_mult,
        eta_min=cfg.min_lr,
    )
