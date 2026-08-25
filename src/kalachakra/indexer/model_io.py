"""Model + geometry loading and batched inference for the Great Indexer.

PyTorch is used strictly as the physics engine: load the trained VQ v3 checkpoint,
rebuild its geodesic grid and k-NN neighbour graph, and expose batched routines to
turn daily ephemeris fields into (a) continuous pre-quantization latents and (b)
discrete token ids. Everything else in the pipeline is numpy / DuckDB.
"""

from __future__ import annotations

import numpy as np


def select_device(pref: str = ""):
    import torch
    if pref:
        return torch.device(pref)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model_and_grid(checkpoint: str, device):
    """Load a v3 checkpoint -> (model, cfg, grid, neighbors)."""
    import torch

    from ..grid.geodesic import Grid
    from ..models.autoencoder_v3 import VQAutoencoderV3, VQAutoencoderV3Config

    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if ck.get("format") != "kalachakra-vqmodel-v3":
        raise ValueError(f"not a v3 checkpoint (format={ck.get('format')!r}): {checkpoint}")
    cfg = VQAutoencoderV3Config(**ck["config"])
    neighbors = np.asarray(ck["neighbors"], dtype=np.int64)
    model = VQAutoencoderV3(cfg, neighbors)
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    xyz = np.asarray(ck["grid_xyz"], dtype=np.float64)
    lat = np.arcsin(np.clip(xyz[:, 2], -1.0, 1.0))
    lon = np.arctan2(xyz[:, 1], xyz[:, 0])
    grid = Grid(xyz=xyz, lat=lat, lon=lon)
    return model, cfg, grid, neighbors


def project_fields(grid, jds) -> np.ndarray:
    """Local topocentric field for a list of Julian Days -> ``(len(jds), N, 50)``."""
    from ..ephemeris import global_state
    from ..projection import spatial

    out = []
    for j in jds:
        g = global_state.global_state_frame(float(j))
        out.append(spatial.project(g, float(j), grid).reshape(grid.n_nodes, -1))
    return np.stack(out, axis=0)


def tokenize_batch(model, fields: np.ndarray, device, want_latent_norm: bool = False):
    """Token ids (and optionally pre-quant ||z||) for a batch of daily fields.

    ``fields`` is ``(B, N, 50)``. Returns ``tokens (B, N) int64`` and, when
    requested, ``node_mag (B, N) float64`` = the L2 norm of the continuous latent
    immediately before the quantization bottleneck (the archetype's tension
    intensity, which the unit-normalized cosine codebook does not carry).
    """
    import torch

    e = torch.from_numpy(fields[:, None].astype(np.float32)).to(device)   # (B,1,N,50)
    with torch.no_grad():
        z = model.encode(e)                       # (B,1,N,64) continuous, pre-VQ
        _zq, idx, _l, _p = model.vq(z)            # eval() -> no EMA mutation
        tokens = idx[:, 0].detach().cpu().numpy().astype(np.int64)
        if want_latent_norm:
            mag = z[:, 0].norm(dim=2)             # (B, N) pre-quant L2 norm
            return tokens, mag.detach().cpu().numpy().astype(np.float64)
    return tokens, None


def auto_node_batch(n_nodes: int, override: int = 0) -> int:
    """Timeline steps per encoder forward that keep a full-mesh activation bounded."""
    if override and override > 0:
        return override
    return int(max(1, min(64, 1_500_000 // max(n_nodes, 1))))
