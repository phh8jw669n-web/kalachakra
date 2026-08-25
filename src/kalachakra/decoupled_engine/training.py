"""Self-supervised training loop for the decoupled engine.

Composes the three physics losses (geometric interference contrastive, terrestrial
geodesic smoothness, temporal continuity), optimises both models jointly with
AdamW + cosine-annealing-with-warmup + gradient clipping, and checkpoints weights,
optimizer/scheduler state and metrics. Runs on CUDA, Apple-Silicon MPS or CPU with
automatic mixed precision.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch

from .. import geometry as geo
from ..ephemeris import global_state as gs
from .bundle import build_models, save_checkpoint
from .config import EngineConfig
from .dataset import build_dataloader, move_batch
from .features import geodesic_neighbor
from .losses import (
    culmination_edge_permission,
    geometric_interference_contrastive_loss,
    temporal_continuity_loss,
    terrestrial_smoothness_loss,
)


def select_device(pref: str = "") -> torch.device:
    if pref:
        return torch.device(pref)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def cosine_warmup(optimizer, warmup_steps: int, max_steps: int):
    """Linear warmup then cosine anneal to zero over ``max_steps``."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def composite_step(sky, earth, batch, cfg: EngineConfig, device):
    """Run both models on one batch and return ``(total_loss, parts_dict)``."""
    cel, jds, coords = batch                                   # (B,T,10,5),(B,T),(B,P,2)
    b, t = cel.shape[0], cel.shape[1]
    p = coords.shape[1]
    flat_cel = cel.reshape(b * t, cfg.sky.n_bodies, cfg.sky.in_features)
    z = sky(flat_cel)                                         # (B*T, 512)

    coords_bt = coords.unsqueeze(1).expand(b, t, p, 2).reshape(b * t, p, 2)
    color = earth(z, coords_bt)                              # (B*T, P, 3)

    # 1) geometric interference contrastive loss (Sky Encoder latents)
    l_geo = geometric_interference_contrastive_loss(
        z, flat_cel, temperature=cfg.train.geo_temperature)

    # 2) terrestrial geodesic-smoothness loss (relaxed at culmination boundaries)
    eps_rad = math.radians(cfg.train.geodesic_eps_deg)
    coords_nb = geodesic_neighbor(coords_bt, eps_rad)
    color_nb = earth(z, coords_nb)
    gmst_deg = geo.greenwich_mean_sidereal_time_deg(
        jds.reshape(-1).detach().cpu().numpy())
    gmst_rad = torch.as_tensor(np.deg2rad(gmst_deg), dtype=color.dtype, device=device)
    permission = culmination_edge_permission(flat_cel, coords_bt, gmst_rad)
    l_terr = terrestrial_smoothness_loss(color, color_nb, eps_rad, permission)

    # 3) temporal continuity loss (second time-difference of colour)
    l_temp = temporal_continuity_loss(color.reshape(b, t, p, 3))

    total = (cfg.train.w_geometric * l_geo
             + cfg.train.w_terrestrial * l_terr
             + cfg.train.w_temporal * l_temp)
    parts = {"geometric": float(l_geo.detach()),
             "terrestrial": float(l_terr.detach()),
             "temporal": float(l_temp.detach()),
             "total": float(total.detach())}
    return total, parts


def train(cfg: EngineConfig, *, num_workers: int = 0, ephe_path: str | None = None,
          jpl_file: str | None = None, max_steps: int | None = None,
          logger=print) -> Path:
    """Train both models and return the path to the final checkpoint."""
    if not gs.ephemeris_available():
        raise RuntimeError("pyswisseph is required for training the decoupled engine.")
    gs.configure_from_args(ephe_path=ephe_path, jpl_file=jpl_file)

    torch.manual_seed(cfg.train.seed)
    device = select_device(cfg.train.device)
    sky, earth = build_models(cfg)
    sky, earth = sky.to(device), earth.to(device)
    steps_target = max_steps if max_steps is not None else cfg.train.max_steps

    params = list(sky.parameters()) + list(earth.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg.train.lr,
                                  weight_decay=cfg.train.weight_decay)
    scheduler = cosine_warmup(optimizer, cfg.train.warmup_steps, steps_target)
    use_amp = cfg.train.amp and device.type in ("cuda", "mps")
    scaler = torch.amp.GradScaler(enabled=cfg.train.amp and device.type == "cuda")

    out_dir = Path(cfg.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger(f"decoupled-train on {device} | tension={cfg.sky.tension_dim} "
           f"| bodies={cfg.sky.n_bodies} | target {steps_target} steps")

    step, t0, last = 0, time.time(), {}
    sky.train()
    earth.train()
    epoch = 0
    while step < steps_target:
        loader = build_dataloader(cfg.data, cfg.train.batch_size,
                                  num_workers=num_workers, device=device, epoch=epoch)
        for batch in loader:
            if num_workers > 0:
                batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=use_amp):
                total, parts = composite_step(sky, earth, batch, cfg, device)
            if scaler.is_enabled():
                scaler.scale(total).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, cfg.train.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                total.backward()
                torch.nn.utils.clip_grad_norm_(params, cfg.train.grad_clip)
                optimizer.step()
            scheduler.step()
            step += 1
            last = parts
            if step % cfg.train.log_every == 0 or step == 1:
                lr = scheduler.get_last_lr()[0]
                rate = step / max(time.time() - t0, 1e-9)
                logger(f"  step {step:5d}/{steps_target}  total={parts['total']:.4f}  "
                       f"geo={parts['geometric']:.4f}  terr={parts['terrestrial']:.4f}  "
                       f"temp={parts['temporal']:.4f}  lr={lr:.2e}  ({rate:.2f} it/s)")
            if step % cfg.train.save_every == 0:
                save_checkpoint(out_dir / f"step_{step:06d}.pt", sky, earth, cfg, step,
                                optimizer=optimizer, scheduler=scheduler, metrics=parts)
            if step >= steps_target:
                break
        epoch += 1

    final = save_checkpoint(out_dir / "model_final.pt", sky, earth, cfg, step,
                            optimizer=optimizer, scheduler=scheduler, metrics=last)
    logger(f"done: {step} steps in {time.time() - t0:.1f}s -> {final}")
    return final
