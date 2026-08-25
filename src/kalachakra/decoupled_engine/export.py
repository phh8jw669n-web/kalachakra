"""Export the trained decoupled models to TorchScript (and ONNX when available).

TorchScript is always produced (no extra dependency). ONNX is attempted only if
the ``onnx`` package is importable, with dynamic axes so the exported Earth Lens
still evaluates an arbitrary number of coordinates.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .bundle import load_checkpoint


def _examples(cfg, device):
    b, p = 2, 16
    sky_in = torch.randn(b, cfg.sky.n_bodies, cfg.sky.in_features, device=device)
    tension = torch.randn(b, cfg.sky.tension_dim, device=device)
    latlon = torch.randn(b, p, 2, device=device)
    return sky_in, tension, latlon


def export_torchscript(sky, earth, cfg, out_dir, device="cpu") -> dict:
    """Trace both models to TorchScript; returns the written paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sky, earth = sky.to(device).eval(), earth.to(device).eval()
    sky_in, tension, latlon = _examples(cfg, device)
    with torch.no_grad():
        sky_ts = torch.jit.trace(sky, sky_in)
        earth_ts = torch.jit.trace(earth, (tension, latlon))
    sky_path = out / "sky_encoder.ts.pt"
    earth_path = out / "earth_lens.ts.pt"
    sky_ts.save(str(sky_path))
    earth_ts.save(str(earth_path))
    return {"sky": str(sky_path), "earth": str(earth_path)}


def export_onnx(sky, earth, cfg, out_dir, device="cpu") -> dict:
    """Export both models to ONNX (requires the ``onnx`` package)."""
    try:
        import onnx  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("ONNX export needs `pip install onnx`.") from exc
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sky, earth = sky.to(device).eval(), earth.to(device).eval()
    sky_in, tension, latlon = _examples(cfg, device)
    sky_path = out / "sky_encoder.onnx"
    earth_path = out / "earth_lens.onnx"
    torch.onnx.export(
        sky, (sky_in,), str(sky_path), input_names=["celestial"],
        output_names=["tension"],
        dynamic_axes={"celestial": {0: "batch"}, "tension": {0: "batch"}},
        opset_version=17)
    torch.onnx.export(
        earth, (tension, latlon), str(earth_path),
        input_names=["tension", "latlon"], output_names=["oklab"],
        dynamic_axes={"tension": {0: "batch"}, "latlon": {0: "batch", 1: "points"},
                      "oklab": {0: "batch", 1: "points"}},
        opset_version=17)
    return {"sky": str(sky_path), "earth": str(earth_path)}


def export_from_checkpoint(ckpt_path, out_dir, fmt: str = "torchscript",
                           device: str = "cpu") -> dict:
    """Load a checkpoint and export to ``torchscript`` and/or ``onnx``."""
    sky, earth, cfg, _payload = load_checkpoint(ckpt_path, map_location=device)
    written: dict = {}
    if fmt in ("torchscript", "both"):
        written["torchscript"] = export_torchscript(sky, earth, cfg, out_dir, device)
    if fmt in ("onnx", "both"):
        written["onnx"] = export_onnx(sky, earth, cfg, out_dir, device)
    return written
