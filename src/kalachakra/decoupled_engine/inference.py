"""Inference engine bridging the trained decoupled models to the frontend.

Two evaluation modes:

* :meth:`DecoupledInference.global_texture` -- run the Sky Encoder once for a
  timestamp, then evaluate the Earth Lens across a full equirectangular grid and
  return an OKLab array plus a gamma-sRGB byte buffer for direct WebGL texture
  upload.
* :meth:`DecoupledInference.pinpoint` -- a single (lat, lon) query returning the
  perceptual colour and the per-planet attention attribution breakdown.

The Sky Encoder is evaluated exactly once per timestamp (the tension vector is
shared by every terrestrial query), which is what makes on-demand, infinite-
resolution spatial evaluation cheap.
"""

from __future__ import annotations

import numpy as np
import torch

from ..ephemeris import global_state as gs
from ..ephemeris.calendar import parse_datetime
from .bundle import load_checkpoint
from .color import oklab_to_srgb8
from .features import BODY_NAMES, celestial_features, equirect_grid
from .sky_encoder import SkyEncoder


def select_device(pref: str = "") -> torch.device:
    if pref:
        return torch.device(pref)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def jd_from_timestamp(ts) -> float:
    """Accept a Julian Day float or an ISO-8601 / 'now' string -> Julian Day."""
    if isinstance(ts, (int, float)):
        return float(ts)
    return float(parse_datetime(str(ts)))


class DecoupledInference:
    """Loads the trained models and answers texture / pinpoint queries."""

    def __init__(self, sky: SkyEncoder, earth, cfg, device: str = ""):
        self.device = select_device(device)
        self.sky = sky.to(self.device).eval()
        self.earth = earth.to(self.device).eval()
        self.cfg = cfg

    @classmethod
    def from_checkpoint(cls, path, device: str = "",
                        ephe_path: str | None = None, jpl_file: str | None = None):
        dev = select_device(device)
        sky, earth, cfg, _payload = load_checkpoint(path, map_location=dev)
        if gs.ephemeris_available():
            gs.configure_from_args(ephe_path=ephe_path, jpl_file=jpl_file)
        return cls(sky, earth, cfg, device=str(dev))

    # -- shared: one Sky Encoder pass per timestamp --------------------------
    @torch.no_grad()
    def tension_vector(self, jd: float) -> torch.Tensor:
        cel = celestial_features(float(jd))                         # (10, 5) numpy
        cel_t = torch.as_tensor(cel, device=self.device).unsqueeze(0)   # (1,10,5)
        return self.sky(cel_t)                                      # (1, 512)

    # -- mode 1: dense global texture ----------------------------------------
    @torch.no_grad()
    def global_texture(self, timestamp, width: int = 512, height: int = 256,
                       chunk: int = 65536) -> dict:
        jd = jd_from_timestamp(timestamp)
        tension = self.tension_vector(jd)                          # (1,512)
        grid = torch.as_tensor(equirect_grid(width, height), device=self.device)
        out = torch.empty((grid.shape[0], self.cfg.earth.out_channels),
                          device=self.device)
        for s in range(0, grid.shape[0], chunk):
            pts = grid[s:s + chunk].unsqueeze(0)                   # (1, n, 2)
            out[s:s + pts.shape[1]] = self.earth(tension, pts)[0]
        oklab = out.reshape(height, width, -1).float().cpu().numpy()
        rgb8 = oklab_to_srgb8(oklab)
        return {"jd": jd, "width": width, "height": height,
                "oklab": oklab, "rgb8": rgb8, "bytes": rgb8.tobytes()}

    # -- mode 2: pinpoint coordinate query -----------------------------------
    @torch.no_grad()
    def pinpoint(self, timestamp, lat_deg: float, lon_deg: float) -> dict:
        jd = jd_from_timestamp(timestamp)
        cel = celestial_features(float(jd))
        cel_t = torch.as_tensor(cel, device=self.device).unsqueeze(0)
        tension = self.sky(cel_t)
        latlon = torch.deg2rad(torch.tensor([[lat_deg, lon_deg]], device=self.device))
        oklab = self.earth(tension, latlon.unsqueeze(0))[0, 0].float().cpu().numpy()
        attribution = self.sky.planetary_attribution(cel_t)[0].float().cpu().numpy()
        rgb8 = oklab_to_srgb8(oklab)
        return {
            "jd": jd, "lat": lat_deg, "lon": lon_deg,
            "oklab": oklab, "rgb8": rgb8,
            "attribution": {name: float(w) for name, w in zip(BODY_NAMES, attribution)},
        }

    # -- helper for latent similarity search ---------------------------------
    @torch.no_grad()
    def tension_batch(self, jds) -> np.ndarray:
        """Tension vectors for many timestamps -> ``(len(jds), 512)`` numpy array."""
        vecs = [self.tension_vector(float(j))[0].float().cpu().numpy() for j in jds]
        return np.stack(vecs, axis=0)
