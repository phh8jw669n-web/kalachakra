"""Training loop + weight export for the version10 Topocentric Self-Attention engine."""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch

from .attention import build_model
from .config import V10Config
from .dataset import build_dataloader
from .losses import anchor_loss, color_stats, isometric_loss, isometric_pair_loss
from .state import N_LOCAL, target_features

CHECKPOINT_FORMAT = "kalachakra-version10-topoattn"


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
    logger = logging.getLogger("version10.train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    for h in (logging.StreamHandler(), logging.FileHandler(out_dir / "train_v9.log")):
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
    cfg = V10Config.from_dict(payload["config"])
    model = build_model(**cfg.to_dict()["attn"])
    model.load_state_dict(payload["state_dict"])
    return model, payload, cfg


def train(cfg: V10Config, *, resume: str | None = None, max_steps: int | None = None,
          logger: logging.Logger | None = None) -> Path:
    out_dir = Path(cfg.train.out_dir)
    logger = logger or setup_logger(out_dir)
    torch.manual_seed(cfg.train.seed)
    device = select_device(cfg.train.device)
    steps_target = max_steps if max_steps is not None else cfg.train.max_steps

    model = build_model(**cfg.to_dict()["attn"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    # Weight decay only on >=2-D transform matrices; never on biases, temperatures, the pool
    # query (1-D) or the body-identity embedding (a lookup, not a transform). This lets the decay
    # hold the transforms in the small-number regime (which renders solid lines) without
    # gravitationally squashing the structural parameters.
    decay = [p for n, p in model.named_parameters()
             if p.requires_grad and p.ndim >= 2 and n != "body_emb"]
    no_decay = [p for n, p in model.named_parameters()
                if p.requires_grad and not (p.ndim >= 2 and n != "body_emb")]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.train.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}], lr=cfg.train.lr)
    scheduler = cosine_warmup(optimizer, cfg.train.warmup_steps, steps_target,
                              base_lr=cfg.train.lr, lr_min=cfg.train.lr_min)

    start_step = 0
    if resume:
        payload = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(payload["state_dict"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        start_step = int(payload.get("step", 0))

    a = cfg.attn
    logger.info("=" * 78)
    logger.info(f"KALACHAKRA v10 — TOPOCENTRIC SELF-ATTENTION ({a.n_bodies} tokens: 11 bodies + ASC + MC)")
    logger.info(f"device={device}  params={n_params:,}  steps={start_step}->{steps_target}"
                f"{'  (RESUMED)' if resume else ''}")
    logger.info(f"model: d_model={a.d_model} d_ff={a.d_ff} d_head={a.d_head} blocks={a.n_blocks}"
                f"  head=Cartesian OKLab(a,b) disk<={a.okl_cmax} (render OKLab L={a.okl_l} fixed)")
    isop = (f"iso-pair(fine {cfg.train.tv_weight}@{cfg.train.tv_delta_deg}deg,"
            f" coarse {cfg.train.tv_weight_coarse}@{cfg.train.tv_delta_coarse_deg}deg)"
            if (cfg.train.tv_weight > 0 or cfg.train.tv_weight_coarse > 0) else "iso-pair=off")
    logger.info(f"loss: isometric gamma={cfg.train.gamma}"
                f"  d_sky={cfg.train.w_local}*local+{cfg.train.w_rel}*gated_chord(k={cfg.train.gate_k})"
                f"  anchor={cfg.train.anchor_weight}  {isop}"
                f"  wd={cfg.train.weight_decay} qk_norm={a.qk_norm}(temp<={a.attn_temp_max})"
                f"  AdamW lr={cfg.train.lr}->{cfg.train.lr_min}  batch={cfg.data.batch}")
    logger.info("=" * 78)

    loader = build_dataloader(cfg.data, cfg.train.gate_k, num_workers=cfg.train.num_workers)
    dc = cfg.data
    gk, gamma, wl, wr = cfg.train.gate_k, cfg.train.gamma, cfg.train.w_local, cfg.train.w_rel
    scales = [(cfg.train.tv_delta_deg, cfg.train.tv_weight),
              (cfg.train.tv_delta_coarse_deg, cfg.train.tv_weight_coarse)]
    use_pair = any(w > 0 for _, w in scales)
    tv_rng = np.random.default_rng(cfg.train.seed + 777)
    tv_bs = min(256, cfg.data.batch)
    model.train()
    step = start_step
    t0 = time.time()
    last: dict = {}
    it = iter(loader)
    while step < steps_target:
        (feat,) = next(it)
        feat = feat.to(device)
        x = feat[:, :N_LOCAL]                              # 13x3 tokens (model reshapes)
        optimizer.zero_grad(set_to_none=True)
        color = model(x)
        loss = (isometric_loss(feat, color, gamma, wl, wr)
                + cfg.train.anchor_weight * anchor_loss(color))
        if use_pair:
            # v10.1 anti-winding: enforce the isometric metric at fine + coarse spatial scale.
            # Same base point, neighbours in a random azimuth per scale. The reference is the
            # true sky distance, so this removes winding without dulling a genuine gradient.
            lat = tv_rng.uniform(dc.lat_min, dc.lat_max, tv_bs)
            lon = tv_rng.uniform(dc.lon_min, dc.lon_max, tv_bs)
            tjd = tv_rng.uniform(dc.jd_start, dc.jd_end, tv_bs)
            f0 = torch.from_numpy(target_features(lat, lon, tjd, gk)).to(device)
            c0 = model(f0[:, :N_LOCAL])
            for scale, w in scales:
                if w <= 0:
                    continue
                az = tv_rng.uniform(0.0, 2.0 * math.pi, tv_bs)
                lat2 = np.clip(lat + scale * np.cos(az), -89.9, 89.9)
                lon2 = lon + scale * np.sin(az)
                f1 = torch.from_numpy(target_features(lat2, lon2, tjd, gk)).to(device)
                loss = loss + w * isometric_pair_loss(c0, model(f1[:, :N_LOCAL]), f0, f1,
                                                      gamma, wl, wr)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        optimizer.step()
        scheduler.step()
        step += 1

        if step % cfg.train.log_every == 0 or step == start_step + 1:
            cs = color_stats(color)
            last = {"loss": float(loss.detach()), **cs}
            elapsed = max(time.time() - t0, 1e-9)
            rate = (step - start_step) / elapsed
            logger.info(
                f"step {step:6d}/{steps_target}  loss={loss.detach():.4f}  "
                f"a={cs['mean_a']:+.3f}±{cs['std_a']:.3f}  b={cs['mean_b']:+.3f}±{cs['std_b']:.3f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}  ({rate:.1f} it/s)")
        if step % cfg.train.save_every == 0:
            save_checkpoint(out_dir / f"step_{step:06d}.pt", model, optimizer,
                            scheduler, step, cfg, last)
            logger.info(f"  saved checkpoint at step {step}")

    final = save_checkpoint(out_dir / "model_final.pt", model, optimizer, scheduler,
                            step, cfg, last)
    logger.info(f"done: {step} steps in {time.time() - t0:.1f}s -> {final}")
    return final


def export_weights_json(checkpoint: str, out_path: str) -> str:
    """Dump the attention weights (+ head/target metadata) as JSON for the shader/HUD."""
    model, _payload, cfg = load_checkpoint(checkpoint, map_location="cpu")
    model.eval()
    payload = model.export_weights()
    payload["gamma"] = cfg.train.gamma
    payload["w_local"] = cfg.train.w_local       # provenance only (inference is a plain forward)
    payload["w_rel"] = cfg.train.w_rel
    payload["gate_k"] = cfg.train.gate_k
    payload["tv_weight"] = cfg.train.tv_weight
    payload["tv_weight_coarse"] = cfg.train.tv_weight_coarse
    payload["weight_decay"] = cfg.train.weight_decay
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(payload))
    return out_path
