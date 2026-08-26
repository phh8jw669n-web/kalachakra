"""Inference for the Local Sky Autoencoder.

Loads a trained checkpoint and, given ``(timestamp, lat, lon)``, computes the Local
Sky Matrix, runs the **encoder only** (the decoder is discarded at inference), and
returns the OKLab bottleneck colour, its sRGB8 conversion, and -- optionally -- the
encoder's self-attention weights (which planets drove the local colour).
"""

from __future__ import annotations

import numpy as np
import torch

from ..ephemeris import global_state as gs
from ..ephemeris.calendar import parse_datetime
from .color import oklab_to_srgb8
from .features import BODY_NAMES, local_sky_matrix
from .model import LocalSkyAutoencoder
from .training import load_checkpoint, select_device


def jd_from_timestamp(ts) -> float:
    """Accept a Julian Day (number/numeric string) or an ISO-8601 / 'now' string."""
    if isinstance(ts, (int, float)):
        return float(ts)
    s = str(ts).strip()
    try:
        return float(s)
    except ValueError:
        return float(parse_datetime(s))


class LocalSkyInference:
    """Load a checkpoint and map ``(timestamp, lat, lon)`` -> OKLab colour."""

    def __init__(self, model: LocalSkyAutoencoder, cfg, device: str = ""):
        self.device = select_device(device)
        self.model = model.to(self.device).eval()
        self.cfg = cfg

    @classmethod
    def from_checkpoint(cls, path, device: str = "",
                        ephe_path: str | None = None, jpl_file: str | None = None):
        dev = select_device(device)
        model, _payload, cfg = load_checkpoint(path, map_location=dev)
        if gs.ephemeris_available():
            gs.configure_from_args(ephe_path=ephe_path, jpl_file=jpl_file)
        return cls(model, cfg, device=str(dev))

    @torch.no_grad()
    def infer(self, timestamp, lat: float, lon: float,
              return_attention: bool = True) -> dict:
        """Return ``{oklab_color, srgb8_color, jd, lat, lon[, attention[_raw]]}``."""
        jd = jd_from_timestamp(timestamp)
        feat, _dist = local_sky_matrix(jd, float(lat), float(lon))     # (10,8)
        feat_t = torch.as_tensor(feat, device=self.device).unsqueeze(0)

        if return_attention:
            oklab, attn = self.model.encode(feat_t, return_attention=True)
        else:
            oklab = self.model.encode(feat_t)
        oklab_np = oklab[0].float().cpu().numpy()
        result = {
            "jd": jd, "lat": float(lat), "lon": float(lon),
            "oklab_color": oklab_np.tolist(),
            "srgb8_color": oklab_to_srgb8(oklab_np).tolist(),
        }
        if return_attention:
            # <OBSERVER> token (index 10) attention onto each body, over heads+layers
            obs_to_bodies = attn[0, :, :, -1, :len(BODY_NAMES)].mean(dim=(0, 1))
            w = obs_to_bodies / (obs_to_bodies.sum() + 1e-9)
            result["attention"] = {n: float(v) for n, v in zip(BODY_NAMES, w.cpu())}
            result["attention_raw"] = attn[0].float().cpu().numpy()    # (L,H,11,11)
        return result

    @torch.no_grad()
    def oklab(self, timestamp, lat: float, lon: float) -> np.ndarray:
        """Just the ``(3,)`` OKLab bottleneck (encoder only)."""
        jd = jd_from_timestamp(timestamp)
        feat, _ = local_sky_matrix(jd, float(lat), float(lon))
        feat_t = torch.as_tensor(feat, device=self.device).unsqueeze(0)
        return self.model.encode(feat_t)[0].float().cpu().numpy()
