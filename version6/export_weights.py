#!/usr/bin/env python3
"""Export a trained SIREN checkpoint to the JSON weight payload used by the shader/HUD.

Also emits a small ``golden.json`` (a few (lat,lon,jd) points with their 33-D tensor and
the network's L*a*b* output) so the browser can verify its GLSL/JS ports match the
Python engine bit-for-bit.

Usage:
    python -m version6.export_weights --checkpoint version6/checkpoints/model_final.pt \
        --out version6/web/weights.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def export(checkpoint: str, out_path: str) -> str:
    import numpy as np
    import torch

    from version6 import ephemeris as ephem
    from version6.training import export_weights_json, load_checkpoint

    out_path = export_weights_json(checkpoint, out_path)
    print(f"exported SIREN weights -> {out_path}")

    # golden vector: raw 33-D state + network colour for a handful of points
    model, _payload, _cfg = load_checkpoint(checkpoint, map_location="cpu")
    model.eval()
    pts = [(48.8566, 2.3522, 2468579.123456), (51.5074, -0.1278, 2451545.0),
           (-33.8688, 151.2093, 2500000.5), (0.0, 0.0, 2440000.0),
           (78.2, 15.6, 2460000.0)]
    lat = np.array([p[0] for p in pts])
    lon = np.array([p[1] for p in pts])
    jd = np.array([p[2] for p in pts])
    sky = ephem.topocentric_tensor(lat, lon, jd)             # [P,33]
    with torch.no_grad():
        lab = model(torch.from_numpy(sky)).numpy()
    golden = {
        "points": [
            {"lat": pts[i][0], "lon": pts[i][1], "jd": pts[i][2],
             "sky": sky[i].astype(float).round(6).tolist(),
             "lab": lab[i].astype(float).round(6).tolist()}
            for i in range(len(pts))
        ],
    }
    gp = Path(out_path).with_name("golden.json")
    gp.write_text(json.dumps(golden, indent=2))
    print(f"wrote golden vector -> {gp}")
    return out_path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default="version6/web/weights.json")
    args = p.parse_args(argv)
    export(args.checkpoint, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
