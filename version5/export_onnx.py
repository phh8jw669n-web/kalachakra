#!/usr/bin/env python3
"""Freeze the trained bottleneck to ONNX for in-browser inference.

Only the **encoder** half is exported — the observer-token Transformer plus the
3-neuron OKLab projection. The training decoder is dropped (it exists only to prove
the colours are sufficient). The batch axis is dynamic so the browser can push an
arbitrary number of geographic points per frame, and constant-folding bakes the
learned ``<OBSERVER>`` token into the graph for faster client-side execution.

Usage:
    python -m version5.export_onnx --checkpoint version5/checkpoints/model_final.pt \
        --out version5/web/model_v5.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def export(checkpoint: str, out_path: str, *, opset: int = 17,
           verify: bool = True) -> str:
    """Export ``checkpoint``'s encoder to ``out_path``; return the path.

    When ``verify`` and ``onnxruntime`` is importable, the ONNX output is checked
    against PyTorch on random input so we *know* the client math will match.
    """
    import numpy as np
    import torch

    from version5.model import SkyEnergyEncoder
    from version5.training import load_checkpoint

    model, _payload, cfg = load_checkpoint(checkpoint, map_location="cpu")
    encoder = SkyEnergyEncoder(cfg.model)
    encoder.load_state_dict(model.encoder.state_dict())
    encoder.eval()

    n_bodies, raw = cfg.model.n_bodies, cfg.model.raw_features
    dummy = torch.randn(4, n_bodies, raw)
    out_path = str(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # dynamo=False uses the mature TorchScript exporter (no onnxscript dependency);
    # older torch that predates the ``dynamo`` kwarg exports the same way by default.
    kwargs = dict(
        input_names=["features"], output_names=["oklab"],
        dynamic_axes={"features": {0: "N"}, "oklab": {0: "N"}},
        opset_version=opset, do_constant_folding=True,
    )
    try:
        torch.onnx.export(encoder, dummy, out_path, dynamo=False, **kwargs)
    except TypeError:                                       # torch without ``dynamo``
        torch.onnx.export(encoder, dummy, out_path, **kwargs)
    print(f"exported ONNX -> {out_path}  (input [N,{n_bodies},{raw}] -> output [N,3])")

    try:
        import onnx
        onnx.checker.check_model(onnx.load(out_path))
        print("onnx.checker: model is well-formed")
    except ImportError:
        print("onnx not installed; skipped structural check")

    if verify:
        try:
            import onnxruntime as ort
        except ImportError:
            print("onnxruntime not installed; skipped numerical parity check")
            return out_path
        sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
        probe = torch.randn(37, n_bodies, raw)
        with torch.no_grad():
            ref = encoder(probe).numpy()
        got = sess.run(["oklab"], {"features": probe.numpy()})[0]
        max_err = float(np.max(np.abs(ref - got)))
        print(f"parity check (PyTorch vs onnxruntime): max abs err = {max_err:.2e}")
        assert max_err < 1e-4, "ONNX output diverges from PyTorch!"
        print("dynamic batch verified (exported at N=4, ran at N=37)")

    _write_golden(encoder, cfg, Path(out_path).with_name("golden.json"))
    return out_path


def _write_golden(encoder, cfg, path: Path) -> None:
    """Emit a golden vector so the browser can prove its JS math == the server's.

    Records one timestamp's telemetry and, for a handful of observers, the exact
    ``[10,5]`` features and the encoder's OKLab. ``version5/web/main.js`` rebuilds the
    features from the telemetry with its own spherical math and asserts a match on
    load — the concrete form of the PRD's "client math must match server math" test.
    """
    import json

    import numpy as np
    import torch

    try:
        from version5 import ephemeris as ephem
        from version5 import sky_math
        ephem.configure()
    except Exception as exc:  # noqa: BLE001 - any failure just skips the optional golden
        print(f"golden vector skipped ({exc})")
        return

    jd = 2451545.0                                          # J2000.0, a stable anchor
    tel = ephem.telemetry(jd)
    eq = ephem.equatorial_state(jd)
    gast = ephem.gast_radians(jd)
    pts = [(51.5, -0.12), (-33.9, 151.2), (0.0, 0.0), (78.2, 15.6), (-89.0, 120.0)]
    lat = np.deg2rad([p[0] for p in pts])
    lon = np.deg2rad([p[1] for p in pts])
    feats = sky_math.local_features(eq, gast, lat, lon)     # [P,10,5]
    with torch.no_grad():
        oklab = encoder(torch.from_numpy(feats)).numpy()
    record = {
        "jd": jd, "telemetry": tel,
        "n_bodies": cfg.model.n_bodies, "raw_features": cfg.model.raw_features,
        "points": [
            {"lat_deg": pts[i][0], "lon_deg": pts[i][1],
             "features": feats[i].astype(float).round(6).tolist(),
             "oklab": oklab[i].astype(float).round(6).tolist()}
            for i in range(len(pts))
        ],
    }
    path.write_text(json.dumps(record, indent=2))
    print(f"wrote golden vector -> {path}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="trained .pt checkpoint")
    p.add_argument("--out", default="version5/web/model_v5.onnx",
                   help="output .onnx path (default: served by the frontend)")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--no-verify", action="store_true",
                   help="skip the onnxruntime parity check")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    export(args.checkpoint, args.out, opset=args.opset, verify=not args.no_verify)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
