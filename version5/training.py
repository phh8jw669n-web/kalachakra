"""Training loop for the Sky-Energy Autoencoder.

Standard PyTorch on MPS / CUDA / CPU: AdamW + cosine-warmup, gradient clipping,
resumable checkpoints, and per-step logging of loss + OKLab health (to catch mode
collapse early). ``select_device`` and ``cosine_warmup`` are imported from the root
autoencoder; only the version5-specific loop lives here.
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path

import torch

from kalachakra.local_autoencoder.training import select_device  # reuse device pick

from . import ephemeris as ephem
from .config import V5Config
from .dataset import build_dataloader
from .losses import mass_weights, oklab_stats, reconstruction_loss
from .model import build_model

CHECKPOINT_FORMAT = "kalachakra-version5"


def cosine_warmup(optimizer, warmup_steps: int, max_steps: int,
                  base_lr: float, lr_min: float = 1e-6):
    """Linear warmup then cosine decay to a non-zero floor ``lr_min``.

    The schedule multiplies the base LR: it ramps ``0 -> 1`` over ``warmup_steps``,
    then cosine-decays ``1 -> lr_min/base_lr`` so the learning rate lands exactly on
    ``lr_min`` at ``max_steps`` (not 0). Warmup defaults to 1,000 steps in the config.
    """
    floor = (lr_min / base_lr) if base_lr > 0 else 0.0

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        cos = 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))      # 1 -> 0
        return floor + (1.0 - floor) * cos                          # 1 -> floor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def setup_logger(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("version5.train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    for h in (logging.StreamHandler(), logging.FileHandler(out_dir / "train_v5.log")):
        h.setFormatter(fmt)
        logger.addHandler(h)
    logger.propagate = False
    return logger


def save_checkpoint(path: Path, model, optimizer, scheduler, step: int,
                    cfg: V5Config, metrics: dict | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT, "config": cfg.to_dict(), "step": int(step),
        "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "metrics": metrics or {},
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    return path


def load_checkpoint(path, map_location="cpu"):
    """Load a checkpoint -> ``(model, payload, cfg)`` with weights restored."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    cfg = V5Config.from_dict(payload["config"])
    model = build_model(cfg.model)
    model.load_state_dict(payload["state_dict"])
    return model, payload, cfg


def train(cfg: V5Config, *, resume: str | None = None, max_steps: int | None = None,
          ephe_path: str | None = None, jpl_file: str | None = None,
          logger: logging.Logger | None = None) -> Path:
    """Train the autoencoder; returns the final checkpoint path."""
    backend = ephem.configure(ephe_path=ephe_path, jpl_file=jpl_file)

    out_dir = Path(cfg.train.out_dir)
    logger = logger or setup_logger(out_dir)
    torch.manual_seed(cfg.train.seed)
    device = select_device(cfg.train.device)
    steps_target = max_steps if max_steps is not None else cfg.train.max_steps

    model = build_model(cfg.model).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                                  weight_decay=cfg.train.weight_decay)
    scheduler = cosine_warmup(optimizer, cfg.train.warmup_steps, steps_target,
                              base_lr=cfg.train.lr, lr_min=cfg.train.lr_min)
    body_w = mass_weights() if cfg.train.mass_weighting else None

    start_step = 0
    if resume:
        payload = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(payload["state_dict"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        start_step = int(payload.get("step", 0))

    use_amp = cfg.train.amp and device.type in ("cuda", "mps")
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float16

    logger.info("=" * 78)
    logger.info("SKY-ENERGY AUTOENCODER (version5)")
    logger.info(f"device={device}  params={n_params:,}  steps={start_step}->{steps_target}"
                f"{'  (RESUMED)' if resume else ''}")
    logger.info(f"model: d_model={cfg.model.d_model} heads={cfg.model.nhead} "
                f"layers={cfg.model.num_layers} ff={cfg.model.dim_feedforward} "
                f"pool={cfg.model.pool}")
    logger.info(f"data: jd[{cfg.data.start_jd:.1f}..{cfg.data.end_jd:.1f}] "
                f"ephemeris={backend} locations/step={cfg.data.locations_per_step} "
                f"workers={cfg.train.num_workers}")
    logger.info(f"loss: mass_w={cfg.train.mass_weighting} "
                f"obs_weight={cfg.train.obs_weight} (per-token MSE, equal bodies)")
    logger.info(f"optim: AdamW lr={cfg.train.lr}->{cfg.train.lr_min} "
                f"wd={cfg.train.weight_decay} warmup={cfg.train.warmup_steps} "
                f"cosine amp={use_amp}")
    logger.info("=" * 78)

    loader = build_dataloader(cfg.data, num_workers=cfg.train.num_workers,
                              ephe_path=ephe_path, jpl_file=jpl_file,
                              pin_memory=(device.type == "cuda"))
    model.train()
    step = start_step
    t0 = time.time()
    last: dict = {}
    it = iter(loader)
    while step < steps_target:
        feats, obs, target, _jd = next(it)
        feats = feats.to(device)
        obs = obs.to(device)
        target = target.to(device)
        # observer reconstruction target: (sin,cos) of Asc/MC/Vertex  -> [B,3,2]
        obs_target = torch.stack([torch.sin(obs), torch.cos(obs)], dim=-1)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            recon_body, recon_obs, oklab = model(feats, obs)
            loss = reconstruction_loss(recon_body.float(), target.float(),
                                       recon_obs.float(), obs_target.float(),
                                       obs_weight=cfg.train.obs_weight, body_w=body_w)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        optimizer.step()
        scheduler.step()
        step += 1

        if step % cfg.train.log_every == 0 or step == start_step + 1:
            st = oklab_stats(oklab)
            last = {"loss": float(loss.detach()), **st}
            elapsed = max(time.time() - t0, 1e-9)
            rate = (step - start_step) / elapsed
            samp = rate * cfg.data.locations_per_step
            collapse = " ** COLLAPSE? **" if (st["mean_chroma"] < 1e-3
                                              and st["std_L"] < 1e-3) else ""
            logger.info(
                f"step {step:6d}/{steps_target}  loss={loss.detach():.5f}  "
                f"L={st['mean_L']:.3f}±{st['std_L']:.3f}  chroma={st['mean_chroma']:.4f}  "
                f"|a|={st['mean_abs_a']:.3f} |b|={st['mean_abs_b']:.3f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}  "
                f"({rate:.1f} it/s, {samp:,.0f} samp/s){collapse}")
        if step % cfg.train.save_every == 0:
            save_checkpoint(out_dir / f"step_{step:06d}.pt", model, optimizer,
                            scheduler, step, cfg, last)
            logger.info(f"  saved checkpoint at step {step}")

    final = save_checkpoint(out_dir / "model_final.pt", model, optimizer, scheduler,
                            step, cfg, last)
    logger.info(f"done: {step} steps in {time.time() - t0:.1f}s -> {final}")
    return final
