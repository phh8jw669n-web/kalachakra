"""Training + export for the version7 regional/city-grid engine.

The model, loss and schedule are reused wholesale from version6 (same isometric metric, same
bounded/soft-clamped L*a*b* head that eliminates neon clipping). Only the data distribution
changes — structured city/grid nodes instead of pure uniform noise. Export emits the three
files the texture-mapping frontend consumes: ``weights.json`` (the SIREN), ``cities.json``
(the metropolitan markers) and ``manifest.json`` (grid + timeline + architecture).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from version6.ephemeris import BODY_NAMES, N_BODIES, STATE_DIM
from version6.losses import anchor_loss, color_stats, isometric_loss
from version6.siren import build_siren
from version6.training import cosine_warmup, select_device, setup_logger

from .cities import unique_cities
from .config import V7Config
from .dataset import build_dataloader

CHECKPOINT_FORMAT = "kalachakra-version7-grid"


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
    cfg = V7Config.from_dict(payload["config"])
    model = build_siren(**cfg.to_dict()["siren"])
    model.load_state_dict(payload["state_dict"])
    return model, payload, cfg


def train(cfg: V7Config, *, resume: str | None = None, max_steps: int | None = None,
          logger=None) -> Path:
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

    uni = max(0.0, 1.0 - cfg.data.city_frac - cfg.data.grid_frac)
    logger.info("=" * 78)
    logger.info("KALACHAKRA v7 — REGIONAL / CITY-GRID SIREN")
    logger.info(f"device={device}  params={n_params:,}  steps={start_step}->{steps_target}"
                f"{'  (RESUMED)' if resume else ''}")
    logger.info(f"siren: in={cfg.siren.in_features} hidden={cfg.siren.hidden}x"
                f"{cfg.siren.hidden_layers} out={cfg.siren.out_features} omega0={cfg.siren.omega0}"
                f"  Lab-head=(L 0..100, ab +/-{cfg.siren.lab_ab:.0f})")
    logger.info(f"data: cities={cfg.data.city_frac:.0%} grid={cfg.data.grid_frac:.0%} "
                f"uniform={uni:.0%}  grid_step={cfg.data.grid_step_deg}deg jitter={cfg.data.jitter_deg}deg "
                f"batch={cfg.data.batch}")
    logger.info(f"render grid: {cfg.grid.width}x{cfg.grid.height}  "
                f"jd[{cfg.data.jd_start:.0f}..{cfg.data.jd_end:.0f}]")
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
                f"lr={scheduler.get_last_lr()[0]:.2e}  ({rate:.1f} it/s)")
        if step % cfg.train.save_every == 0:
            save_checkpoint(out_dir / f"step_{step:06d}.pt", model, optimizer,
                            scheduler, step, cfg, last)
            logger.info(f"  saved checkpoint at step {step}")

    final = save_checkpoint(out_dir / "model_final.pt", model, optimizer, scheduler,
                            step, cfg, last)
    logger.info(f"done: {step} steps in {time.time() - t0:.1f}s -> {final}")
    return final


# ---------------------------------------------------------------------------
# export for the texture-mapping frontend
# ---------------------------------------------------------------------------
def export_weights_json(checkpoint: str, out_path: str) -> str:
    """Dump the SIREN weights (+ bounded-head metadata) as JSON for the field worker."""
    model, _payload, cfg = load_checkpoint(checkpoint, map_location="cpu")
    model.eval()
    payload = model.export_weights()
    payload["color_scale"] = cfg.train.color_scale
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(payload))
    return out_path


def export_manifest(cfg: V7Config, out_dir: str) -> dict:
    """Write ``cities.json`` and ``manifest.json`` next to ``weights.json``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cities = [{"name": n, "lat": lat, "lon": lon, "region": reg}
              for (n, lat, lon, reg) in unique_cities()]
    (out / "cities.json").write_text(json.dumps(cities))
    manifest = {
        "format": CHECKPOINT_FORMAT,
        "grid": {"width": cfg.grid.width, "height": cfg.grid.height},
        "timeline": {"jd_start": cfg.data.jd_start, "jd_end": cfg.data.jd_end},
        "state_dim": STATE_DIM, "n_bodies": N_BODIES, "bodies": list(BODY_NAMES),
        "siren": {"hidden": cfg.siren.hidden, "hidden_layers": cfg.siren.hidden_layers,
                  "omega0": cfg.siren.omega0},
        "n_cities": len(cities),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
