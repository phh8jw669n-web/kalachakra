#!/usr/bin/env python3
"""Export a trained version7 checkpoint into the frontend bundle.

Writes ``weights.json`` (the SIREN), ``cities.json`` (metropolitan markers) and
``manifest.json`` (render grid + timeline + architecture) into the target directory — the
three files the texture-mapping frontend loads.

Usage:
    python -m version7.export --checkpoint version7/checkpoints/model_final.pt --out version7/web
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default="version7/web")
    args = p.parse_args(argv)

    from version7.training import export_manifest, export_weights_json, load_checkpoint
    _model, _payload, cfg = load_checkpoint(args.checkpoint, map_location="cpu")
    out = export_weights_json(args.checkpoint, str(Path(args.out) / "weights.json"))
    manifest = export_manifest(cfg, args.out)
    print(f"exported SIREN weights -> {out}")
    print(f"exported cities.json ({manifest['n_cities']} hubs) + manifest.json -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
