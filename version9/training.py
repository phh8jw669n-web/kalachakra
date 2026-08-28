"""Training loop + weight export for the version9 Topocentric Self-Attention engine."""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path

import torch

from .attention import build_model
from .config import V9Config
from .dataset import build_dataloader
from .losses import anchor_loss, color_stats, isometric_loss
from .state import N_LOCAL

CHECKPOINT_FORMAT = "kalachakra-version9-topoattn"


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
    logger = logging.getLogger("version9.train")
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
    cfg = V9Config.from_dict(payload["config"])
    model = build_model(**cfg.to_dict()["attn"])
    model.load_state_dict(payload["state_dict"])
    return model, payload, cfg


def train(cfg: V9Config, *, resume: str | None = None, max_steps: int | None = None,
          logger: logging.Logger | None = None) -> Path:
    out_dir = Path(cfg.train.out_dir)
    logger = logger or setup_logger(out_dir)
    torch.manual_seed(cfg.train.seed)
    device = select_device(cfg.train.device)
    steps_target = max_steps if max_steps is not None else cfg.train.max_steps

    model = build_model(**cfg.to_dict()["attn"]).to(device)
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

    a = cfg.attn
    logger.info("=" * 78)
    logger.info("KALACHAKRA v9 — TOPOCENTRIC SELF-ATTENTION (11 body tokens)")
    logger.info(f"device={device}  params={n_params:,}  steps={start_step}->{steps_target}"
                f"{'  (RESUMED)' if resume else ''}")
    logger.info(f"model: d_model={a.d_model} d_ff={a.d_ff} d_head={a.d_head} blocks={a.n_blocks}"
                f"  head=OKLCH C<={a.okl_cmax} H (render OKLab L={a.okl_l} fixed)")
    logger.info(f"loss: isometric gamma={cfg.train.gamma}"
                f"  d_sky={cfg.train.w_local}*local+{cfg.train.w_rel}*gated_chord(k={cfg.train.gate_k})"
                f"  anchor={cfg.train.anchor_weight}  optim=AdamW lr={cfg.train.lr}->{cfg.train.lr_min}"
                f"  batch={cfg.data.batch}")
    logger.info("=" * 78)

    loader = build_dataloader(cfg.data, cfg.train.gate_k, num_workers=cfg.train.num_workers)
    model.train()
    step = start_step
    t0 = time.time()
    last: dict = {}
    it = iter(loader)
    while step < steps_target:
        (feat,) = next(it)
        feat = feat.to(device)
        x = feat[:, :N_LOCAL]                              # 11x3 tokens (model reshapes)
        optimizer.zero_grad(set_to_none=True)
        color = model(x)
        loss = (isometric_loss(feat, color, cfg.train.gamma,
                               cfg.train.w_local, cfg.train.w_rel)
                + cfg.train.anchor_weight * anchor_loss(color))
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
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(payload))
    return out_path
