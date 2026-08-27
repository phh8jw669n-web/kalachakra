"""Training loop for the version6 SIREN metric field."""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path

import torch

from .config import V6Config
from .dataset import build_dataloader
from .losses import anchor_loss, color_stats, isometric_loss
from .siren import build_siren

CHECKPOINT_FORMAT = "kalachakra-version6-siren"


def select_device(pref: str = "") -> torch.device:
    if pref:
        return torch.device(pref)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def cosine_warmup(optimizer, warmup_steps, max_steps, base_lr, lr_min):
    floor = (lr_min / base_lr) if base_lr > 0 else 0.0

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def setup_logger(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("version6.train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    for h in (logging.StreamHandler(), logging.FileHandler(out_dir / "train_v6.log")):
        h.setFormatter(fmt)
        logger.addHandler(h)
    logger.propagate = False
    return logger


def save_checkpoint(path: Path, model, optimizer, scheduler, step, cfg, metrics=None):
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
    payload = torch.load(path, map_location=map_location, weights_only=False)
    cfg = V6Config.from_dict(payload["config"])
    model = build_siren(**cfg.to_dict()["siren"])
    model.load_state_dict(payload["state_dict"])
    return model, payload, cfg


def train(cfg: V6Config, *, resume: str | None = None, max_steps: int | None = None,
          logger: logging.Logger | None = None) -> Path:
    out_dir = Path(cfg.train.out_dir)
    logger = logger or setup_logger(out_dir)
    torch.manual_seed(cfg.train.seed)
    device = select_device(cfg.train.device)
    steps_target = max_steps if max_steps is not None else cfg.train.max_steps

    model = build_siren(**cfg.to_dict()["siren"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                                  weight_decay=cfg.train.weight_decay)
    scheduler = cosine_warmup(optimizer, cfg.train.warmup_steps, steps_target,
                              base_lr=cfg.train.lr, lr_min=cfg.train.lr_min)

    start_step = 0
    if resume:
        payload = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(payload["state_dict"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        start_step = int(payload.get("step", 0))

    logger.info("=" * 78)
    logger.info("KALACHAKRA v6 — SIREN ISOMETRIC FIELD")
    logger.info(f"device={device}  params={n_params:,}  steps={start_step}->{steps_target}"
                f"{'  (RESUMED)' if resume else ''}")
    logger.info(f"siren: in={cfg.siren.in_features} hidden={cfg.siren.hidden}x"
                f"{cfg.siren.hidden_layers} out={cfg.siren.out_features} omega0={cfg.siren.omega0}")
    logger.info(f"data: jd[{cfg.data.jd_start:.0f}..{cfg.data.jd_end:.0f}] "
                f"batch={cfg.data.batch} workers={cfg.train.num_workers}")
    logger.info(f"loss: isometric color_scale={cfg.train.color_scale} "
                f"anchor={cfg.train.anchor_weight}  optim=AdamW lr={cfg.train.lr}->{cfg.train.lr_min}")
    logger.info("=" * 78)

    loader = build_dataloader(cfg.data, num_workers=cfg.train.num_workers)
    model.train()
    step = start_step
    t0 = time.time()
    last: dict = {}
    it = iter(loader)
    while step < steps_target:
        (sky,) = next(it)
        sky = sky.to(device)
        optimizer.zero_grad(set_to_none=True)
        color = model(sky)
        loss = (isometric_loss(sky, color, cfg.train.color_scale)
                + cfg.train.anchor_weight * anchor_loss(color))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        optimizer.step()
        scheduler.step()
        step += 1

        if step % cfg.train.log_every == 0 or step == start_step + 1:
            st = color_stats(color)
            last = {"loss": float(loss.detach()), **st}
            elapsed = max(time.time() - t0, 1e-9)
            rate = (step - start_step) / elapsed
            logger.info(
                f"step {step:6d}/{steps_target}  loss={loss.detach():.4f}  "
                f"L*={st['mean_L']:.1f}±{st['std_L']:.1f}  "
                f"a*±{st['std_a']:.1f} b*±{st['std_b']:.1f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}  "
                f"({rate:.1f} it/s, {rate * cfg.data.batch:,.0f} skies/s)")
        if step % cfg.train.save_every == 0:
            save_checkpoint(out_dir / f"step_{step:06d}.pt", model, optimizer,
                            scheduler, step, cfg, last)
            logger.info(f"  saved checkpoint at step {step}")

    final = save_checkpoint(out_dir / "model_final.pt", model, optimizer, scheduler,
                            step, cfg, last)
    logger.info(f"done: {step} steps in {time.time() - t0:.1f}s -> {final}")
    return final


def export_weights_json(checkpoint: str, out_path: str) -> str:
    """Load a checkpoint and dump the SIREN weights (+ metadata) as JSON for the shader.

    Also stores ``lab_offset`` — the rigid shift that moves the network's mean output to
    neutral L*=60. The isometric loss is translation-invariant, so this display gauge is
    fixed here (post-hoc) instead of fought during training. The shader/HUD add it before
    the L*a*b*->sRGB conversion.
    """
    import numpy as np

    from . import ephemeris as ephem
    model, _payload, cfg = load_checkpoint(checkpoint, map_location="cpu")
    model.eval()
    rng = np.random.default_rng(0)
    lat = rng.uniform(cfg.data.lat_min, cfg.data.lat_max, 8192)
    lon = rng.uniform(cfg.data.lon_min, cfg.data.lon_max, 8192)
    jd = rng.uniform(cfg.data.jd_start, cfg.data.jd_end, 8192)
    sky = ephem.topocentric_tensor(lat, lon, jd)
    with torch.no_grad():
        mean = model(torch.from_numpy(sky)).mean(dim=0).numpy()
    offset = (np.array([60.0, 0.0, 0.0]) - mean).astype(float).tolist()

    payload = model.export_weights()
    payload["color_scale"] = cfg.train.color_scale
    payload["lab_offset"] = offset                   # displayed L*a*b* = network output + offset
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(payload))
    return out_path
