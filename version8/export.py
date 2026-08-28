#!/usr/bin/env python3
"""Export a trained version8 checkpoint to weights.json (+ a golden parity file).

``golden.json`` lists a few (lat, lon, jd) points with their 88-D state and the network's
L*a*b* output, so the browser can verify its JS/GLSL ports reproduce the Python engine.

Usage:
    python -m version8.export --checkpoint version8/checkpoints/model_final.pt --out version8/web/weights.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def export(checkpoint: str, out_path: str) -> str:
    import numpy as np
    import torch

    from version8 import state as st
    from version8.training import export_weights_json, load_checkpoint

    out_path = export_weights_json(checkpoint, out_path)
    print(f"exported SIREN weights -> {out_path}")

    model, _payload, _cfg = load_checkpoint(checkpoint, map_location="cpu")
    model.eval()
    pts = [(48.8566, 2.3522, 2468579.123456), (51.5074, -0.1278, 2451545.0),
           (-33.8688, 151.2093, 2500000.5), (0.0, 0.0, 2440000.0), (78.2, 15.6, 2460000.0)]
    lat = np.array([p[0] for p in pts])
    lon = np.array([p[1] for p in pts])
    jd = np.array([p[2] for p in pts])
    state = st.topocentric_state(lat, lon, jd)               # [P,88]
    with torch.no_grad():
        lab = model(torch.from_numpy(state)).numpy()
    golden = {"points": [
        {"lat": pts[i][0], "lon": pts[i][1], "jd": pts[i][2],
         "state": state[i].astype(float).round(6).tolist(),
         "lab": lab[i].astype(float).round(6).tolist()} for i in range(len(pts))]}
    gp = Path(out_path).with_name("golden.json")
    gp.write_text(json.dumps(golden, indent=2))
    print(f"wrote golden vector -> {gp}")
    return out_path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default="version8/web/weights.json")
    args = p.parse_args(argv)
    export(args.checkpoint, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
