"""Training loop for the Local Sky Autoencoder (train_v4).

Standard PyTorch loop on MPS / CUDA / CPU with AMP, AdamW + cosine-warmup, gradient
clipping, rich logging (full config, device, parameter count, per-step loss and
OKLab health metrics to catch mode collapse), and resumable checkpoints (weights +
optimizer + scheduler + step).
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path

import torch

from .config import LocalSkyConfig
from .dataset import build_dataloader
from .losses import oklab_stats, physics_weighted_mse
from .model import build_model

CHECKPOINT_FORMAT = "kalachakra-localsky-v4"


def select_device(pref: str = "") -> torch.device:
    if pref:
        return torch.device(pref)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def cosine_warmup(optimizer, warmup_steps: int, max_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def setup_logger(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("local_sky.train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(out_dir / "train_v4.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.propagate = False
    return logger


def save_checkpoint(path: Path, model, optimizer, scheduler, step: int,
                    cfg: LocalSkyConfig, metrics: dict | None = None) -> Path:
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
    cfg = LocalSkyConfig.from_dict(payload["config"])
    model = build_model(cfg.model)
    model.load_state_dict(payload["state_dict"])
    return model, payload, cfg


def train(cfg: LocalSkyConfig, *, resume: str | None = None,
          max_steps: int | None = None, ephe_path: str | None = None,
          jpl_file: str | None = None,
          logger: logging.Logger | None = None) -> Path:
    """Train the Local Sky Autoencoder; returns the final checkpoint path."""
    from ..ephemeris import global_state as gs
    if not gs.ephemeris_available():
        raise RuntimeError("pyswisseph is required for the Local Sky Autoencoder.")
    backend = gs.configure_from_args(ephe_path=ephe_path, jpl_file=jpl_file)

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

    start_step = 0
    if resume:
        payload = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(payload["state_dict"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        start_step = int(payload.get("step", 0))

    use_amp = cfg.train.amp and device.type in ("cuda", "mps")
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float16
    scaler = torch.amp.GradScaler(enabled=cfg.train.amp and device.type == "cuda")

    # -- rich run banner -----------------------------------------------------
    logger.info("=" * 78)
    logger.info("LOCAL SKY AUTOENCODER (train_v4)")
    logger.info(f"device={device}  params={n_params:,}  steps={start_step}->{steps_target}"
                f"{'  (RESUMED from ' + resume + ')' if resume else ''}")
    logger.info(f"model: d_model={cfg.model.d_model} heads={cfg.model.nhead} "
                f"layers={cfg.model.num_layers} ff={cfg.model.dim_feedforward} "
                f"pool={cfg.model.pool} decoder={list(cfg.model.decoder_hidden)}")
    logger.info(f"optim: AdamW lr={cfg.train.lr} wd={cfg.train.weight_decay} "
                f"warmup={cfg.train.warmup_steps} grad_clip={cfg.train.grad_clip} "
                f"amp={use_amp}({amp_dtype if use_amp else '-'})")
    logger.info(f"data: jd[{cfg.data.start_jd:.1f}..{cfg.data.end_jd:.1f}] "
                f"ephemeris={backend} batch={cfg.train.batch_size} "
                f"workers={cfg.train.num_workers} seed={cfg.train.seed}")
    if cfg.train.num_workers == 0:
        logger.info("HINT: data generation is CPU-bound; pass --workers 8-12 to "
                    "parallelise it and keep the GPU fed (biggest speedup).")
    logger.info(f"io: out_dir={out_dir} save_every={cfg.train.save_every} "
                f"log_every={cfg.train.log_every}")
    logger.info("=" * 78)

    loader = build_dataloader(cfg.data, cfg.train.batch_size,
                              num_workers=cfg.train.num_workers, epoch=start_step,
                              ephe_path=ephe_path, jpl_file=jpl_file,
                              pin_memory=(device.type == "cuda"))
    model.train()
    step = start_step
    t0 = time.time()
    last = {}
    for feats, target, weight in loader:
        feats = feats.to(device)
        target = target.to(device)
        weight = weight.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            recon, oklab = model(feats)
            loss = physics_weighted_mse(recon.float(), target.float(), weight.float())
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()
        scheduler.step()
        step += 1

        if step % cfg.train.log_every == 0 or step == start_step + 1:
            st = oklab_stats(oklab)
            last = {"loss": float(loss.detach()), **st}
            lr = scheduler.get_last_lr()[0]
            rate = (step - start_step) / max(time.time() - t0, 1e-9)
            samp = rate * cfg.train.batch_size
            collapse = " ** COLLAPSE? **" if (st["mean_chroma"] < 1e-3
                                              and st["std_L"] < 1e-3) else ""
            logger.info(
                f"step {step:6d}/{steps_target}  loss={loss.detach():.5f}  "
                f"L={st['mean_L']:.3f}±{st['std_L']:.3f}  chroma={st['mean_chroma']:.4f}  "
                f"|a|={st['mean_abs_a']:.3f} |b|={st['mean_abs_b']:.3f}  "
                f"lr={lr:.2e}  ({rate:.1f} it/s, {samp:,.0f} samp/s){collapse}")
        if step % cfg.train.save_every == 0:
            save_checkpoint(out_dir / f"step_{step:06d}.pt", model, optimizer,
                            scheduler, step, cfg, last)
            logger.info(f"  saved checkpoint at step {step}")
        if step >= steps_target:
            break

    final = save_checkpoint(out_dir / "model_final.pt", model, optimizer, scheduler,
                            step, cfg, last)
    logger.info(f"done: {step} steps in {time.time() - t0:.1f}s -> {final}")
    return final
