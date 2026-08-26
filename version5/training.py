"""Training loop for the Sky-Energy Autoencoder.

Standard PyTorch on MPS / CUDA / CPU: AdamW + cosine-warmup, gradient clipping,
resumable checkpoints, and per-step logging of loss + OKLab health (to catch mode
collapse early). ``select_device`` and ``cosine_warmup`` are imported from the root
autoencoder; only the version5-specific loop lives here.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import torch

from kalachakra.local_autoencoder.training import cosine_warmup, select_device  # reuse

from . import ephemeris as ephem
from .config import V5Config
from .dataset import build_dataloader
from .losses import mass_weights, oklab_stats, reconstruction_loss
from .model import build_model

CHECKPOINT_FORMAT = "kalachakra-version5"


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
    scheduler = cosine_warmup(optimizer, cfg.train.warmup_steps, steps_target)
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
                f"workers={cfg.train.num_workers} mass_w={cfg.train.mass_weighting}")
    logger.info(f"optim: AdamW lr={cfg.train.lr} wd={cfg.train.weight_decay} "
                f"warmup={cfg.train.warmup_steps} amp={use_amp}")
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
        feats, target, _jd = next(it)
        feats = feats.to(device)
        target = target.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            recon, oklab = model(feats)
            loss = reconstruction_loss(recon.float(), target.float(), body_w)
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
