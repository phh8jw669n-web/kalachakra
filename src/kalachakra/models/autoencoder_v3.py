"""
Discrete VQ-bottleneck Spatio-Temporal Autoencoder (v3) — fully self-contained.

This module is INDEPENDENT: it imports only numpy and torch and defines its own
geodesic convolution, Fourier (spectral) temporal operator, spatio-temporal block,
and a discrete Vector-Quantizer codebook inserted directly at the latent
bottleneck. It does not import — and is not imported by — any other kalachakra
model file, so it can evolve on its own track without touching the running v1/v2
pipeline.

Architecture (blueprint §4 + a flat VQ-VAE bottleneck):

    E(t,s) --lift--> [ST blocks] --to_latent--> z_e(64)
        --VectorQuantizer(4096x64)--> z_q(64) + token ids + VQ loss
        --from_latent--> [ST blocks] --project--> reconstruction

The quantizer computes nearest-codebook assignment (argmin Euclidean distance),
returns the straight-through-estimated z_q (gradients flow to the encoder), the
discrete token indices, the VQ loss (codebook + beta*commitment), and the batch
codebook perplexity. The codebook is an nn.Parameter, so it is saved in the
model's state_dict automatically.

Full-mesh safe: the spatial/temporal ops and the VQ nearest-neighbour search are
tiled over the node / token axis so no tensor exceeds Metal's INT_MAX limit, and
the encoder/decoder blocks support gradient checkpointing to fit memory.

Requires PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class VQAutoencoderV3Config:
    n_nodes: int = 122_880
    in_features: int = 50          # LOCAL field width (10 bodies x 5)
    hidden: int = 128
    latent: int = 64
    fourier_modes: int = 32
    knn: int = 7
    n_blocks: int = 3
    codebook_size: int = 4096      # discrete archetypes
    commitment_beta: float = 0.25  # beta in the VQ loss
    node_chunk: int = 8192         # nodes per slice in spatial/temporal ops
    vq_chunk: int = 131_072        # tokens per slice in the codebook search
    grad_checkpoint: bool = False


# ---------------------------------------------------------------------------
# Geodesic mesh k-NN (standalone)
# ---------------------------------------------------------------------------
def build_knn(xyz: np.ndarray, k: int) -> np.ndarray:
    """(N, k) indices of each node's k nearest neighbours on the unit sphere."""
    xyz = np.asarray(xyz, dtype=np.float64)
    n = xyz.shape[0]
    k = min(k, n)
    idx = np.empty((n, k), dtype=np.int64)
    chunk = 2048
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        sims = xyz[s:e] @ xyz.T
        idx[s:e] = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    return idx


# ---------------------------------------------------------------------------
# Geodesic convolution (neighbourhood message passing)
# ---------------------------------------------------------------------------
class GeodesicConvV3(nn.Module):
    """Input/output ``(batch, N, C)``; isotropic self + mean-neighbour kernel."""

    def __init__(self, in_channels: int, out_channels: int, neighbors: np.ndarray):
        super().__init__()
        self.register_buffer("neighbors", torch.as_tensor(neighbors, dtype=torch.long))
        self.self_lin = nn.Linear(in_channels, out_channels)
        self.neigh_lin = nn.Linear(in_channels, out_channels)


class SpectralConv1dV3(nn.Module):
    """1D spectral convolution over time; real-stored complex weights, fp32 FFT."""

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels)
        self.weight = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, length = x.shape
        in_dtype = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            xf = x.float()
            x_ft = torch.fft.rfft(xf, dim=-1)
            keep = min(self.modes, x_ft.shape[-1])
            xr, xi = x_ft.real[:, :, :keep], x_ft.imag[:, :, :keep]
            wr = self.weight[:, :, :keep, 0].float()
            wi = self.weight[:, :, :keep, 1].float()
            out_r = torch.einsum("bix,iox->box", xr, wr) - torch.einsum("bix,iox->box", xi, wi)
            out_i = torch.einsum("bix,iox->box", xr, wi) + torch.einsum("bix,iox->box", xi, wr)
            out_ft = torch.zeros(batch, self.out_channels, x_ft.shape[-1],
                                 dtype=torch.cfloat, device=x.device)
            out_ft[:, :, :keep] = torch.complex(out_r, out_i)
            out = torch.fft.irfft(out_ft, n=length, dim=-1)
        return out.to(in_dtype)


class FourierBlockV3(nn.Module):
    """FNO block: spectral path + pointwise (1x1) residual path + activation."""

    def __init__(self, channels: int, modes: int):
        super().__init__()
        self.spectral = SpectralConv1dV3(channels, channels, modes)
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.spectral(x) + self.pointwise(x))


# ---------------------------------------------------------------------------
# Node-chunked application of the spatial / temporal ops (full-mesh safe)
# ---------------------------------------------------------------------------
def _apply_spatial(conv: GeodesicConvV3, x: torch.Tensor, node_chunk: int) -> torch.Tensor:
    b, t, n, ch = x.shape
    xf = x.reshape(b * t, n, ch)
    idx = conv.neighbors
    if n <= node_chunk:
        agg = xf[:, idx, :].mean(dim=2)
    else:
        aggs = [xf[:, idx[s:s + node_chunk], :].mean(dim=2)
                for s in range(0, n, node_chunk)]
        agg = torch.cat(aggs, dim=1)
    y = conv.self_lin(xf) + conv.neigh_lin(agg)
    return y.reshape(b, t, n, -1)


def _apply_temporal(block: FourierBlockV3, x: torch.Tensor, node_chunk: int) -> torch.Tensor:
    b, t, n, ch = x.shape
    if n <= node_chunk:
        y = x.permute(0, 2, 3, 1).reshape(b * n, ch, t)
        y = block(y)
        return y.reshape(b, n, ch, t).permute(0, 3, 1, 2).contiguous()
    outs = []
    for s in range(0, n, node_chunk):
        xs = x[:, :, s:s + node_chunk, :]
        ns = xs.shape[2]
        ys = block(xs.permute(0, 2, 3, 1).reshape(b * ns, ch, t))
        outs.append(ys.reshape(b, ns, ch, t).permute(0, 3, 1, 2))
    return torch.cat(outs, dim=2).contiguous()


class STBlockV3(nn.Module):
    """Spatial geodesic conv then temporal FNO, both node-chunked, with residuals."""

    def __init__(self, channels: int, modes: int, neighbors: np.ndarray, node_chunk: int):
        super().__init__()
        self.spatial = GeodesicConvV3(channels, channels, neighbors)
        self.temporal = FourierBlockV3(channels, modes)
        self.norm = nn.LayerNorm(channels)
        self.act = nn.GELU()
        self.node_chunk = node_chunk

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.act(_apply_spatial(self.spatial, x, self.node_chunk))
        x = x + _apply_temporal(self.temporal, x, self.node_chunk)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Vector Quantizer (discrete codebook at the bottleneck)
# ---------------------------------------------------------------------------
class VectorQuantizer(nn.Module):
    """Flat VQ-VAE quantizer: nearest-codebook assignment + straight-through
    estimator + (codebook + beta*commitment) loss + batch perplexity.

    Codebook is an ``nn.Parameter`` of shape ``(codebook_size, dim)`` so it is
    saved inside ``state_dict``. The nearest-neighbour search is tiled over the
    token axis (``vq_chunk``) and runs in float32 (FFT/argmin precision), so it is
    safe at the full mesh where the flattened token count is ~10^7.
    """

    def __init__(self, dim: int = 64, codebook_size: int = 4096,
                 beta: float = 0.25, vq_chunk: int = 131_072):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.beta = beta
        self.vq_chunk = vq_chunk
        self.codebook = nn.Parameter(torch.randn(codebook_size, dim) * (1.0 / dim))

    @torch.no_grad()
    def _assign(self, flat: torch.Tensor, cb: torch.Tensor) -> torch.Tensor:
        """Nearest codebook index per row (tiled). Drops the constant ||z_e||^2."""
        cb_sq = (cb * cb).sum(dim=1)                       # (K,)
        m = flat.shape[0]
        idx = torch.empty(m, dtype=torch.long, device=flat.device)
        for s in range(0, m, self.vq_chunk):
            chunk = flat[s:s + self.vq_chunk]              # (c, D)
            d = cb_sq[None, :] - 2.0 * (chunk @ cb.t())    # (c, K)
            idx[s:s + self.vq_chunk] = d.argmin(dim=1)
        return idx

    def forward(self, z_e: torch.Tensor):
        """z_e: (..., dim). Returns (z_q_ste, indices, vq_loss, perplexity)."""
        with torch.autocast(device_type=z_e.device.type, enabled=False):
            ze = z_e.float()
            lead = ze.shape[:-1]
            flat = ze.reshape(-1, self.dim)                # (M, D)
            cb = self.codebook.float()                     # (K, D)

            idx = self._assign(flat, cb)                   # (M,)
            z_q = F.embedding(idx, cb)                     # (M, D), grad -> codebook

            # L_VQ = ||sg[z_e] - z_q||^2 + beta * ||z_e - sg[z_q]||^2
            codebook_loss = F.mse_loss(z_q, flat.detach())
            commitment_loss = F.mse_loss(flat, z_q.detach())
            vq_loss = codebook_loss + self.beta * commitment_loss

            # straight-through estimator: forward is z_q, gradient flows to z_e
            z_q_ste = flat + (z_q - flat).detach()

            # batch codebook perplexity = exp(entropy of the assignment histogram)
            counts = torch.bincount(idx, minlength=self.codebook_size).float()
            probs = counts / counts.sum().clamp_min(1.0)
            perplexity = torch.exp(-(probs * (probs + 1e-10).log()).sum())

        z_q_ste = z_q_ste.reshape(*lead, self.dim).to(z_e.dtype)
        indices = idx.reshape(*lead)
        return z_q_ste, indices, vq_loss.to(z_e.dtype), perplexity.to(z_e.dtype)

    @torch.no_grad()
    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        """Codebook vectors for token indices (inference / decoding from tokens)."""
        return F.embedding(indices, self.codebook)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
class VQAutoencoderV3(nn.Module):
    """Encoder -> to_latent -> Vector Quantizer -> from_latent -> Decoder.

    ``forward`` returns ``(reconstruction, z_q, token_indices, vq_loss)`` and
    stashes the last batch perplexity on ``self.last_perplexity`` for logging.
    """

    def __init__(self, cfg: VQAutoencoderV3Config, neighbors: np.ndarray):
        super().__init__()
        self.cfg = cfg
        self.lift = nn.Linear(cfg.in_features, cfg.hidden)
        self.enc_blocks = nn.ModuleList(
            STBlockV3(cfg.hidden, cfg.fourier_modes, neighbors, cfg.node_chunk)
            for _ in range(cfg.n_blocks)
        )
        self.to_latent = nn.Linear(cfg.hidden, cfg.latent)

        self.vq = VectorQuantizer(cfg.latent, cfg.codebook_size,
                                  cfg.commitment_beta, cfg.vq_chunk)

        self.from_latent = nn.Linear(cfg.latent, cfg.hidden)
        self.dec_blocks = nn.ModuleList(
            STBlockV3(cfg.hidden, cfg.fourier_modes, neighbors, cfg.node_chunk)
            for _ in range(cfg.n_blocks)
        )
        self.project = nn.Linear(cfg.hidden, cfg.in_features)
        self.last_perplexity: float = 0.0

    def _run_blocks(self, blocks, x):
        if self.cfg.grad_checkpoint and self.training:
            from torch.utils.checkpoint import checkpoint
            for block in blocks:
                x = checkpoint(block, x, use_reentrant=False)
            return x
        for block in blocks:
            x = block(x)
        return x

    def encode(self, e: torch.Tensor) -> torch.Tensor:
        """Continuous latent z_e (pre-quantization), shape ``(..., latent)``."""
        x = self.lift(e)
        x = self._run_blocks(self.enc_blocks, x)
        return self.to_latent(x)

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        x = self.from_latent(z_q)
        x = self._run_blocks(self.dec_blocks, x)
        return self.project(x)

    def forward(self, e: torch.Tensor):
        z_e = self.encode(e)
        z_q, indices, vq_loss, perplexity = self.vq(z_e)
        recon = self.decode(z_q)
        self.last_perplexity = float(perplexity.detach())
        return recon, z_q, indices, vq_loss

    @torch.no_grad()
    def tokenize(self, e: torch.Tensor) -> torch.Tensor:
        """Inference: discrete token indices for each (t, s) coordinate."""
        self.eval()
        z_e = self.encode(e)
        _zq, indices, _l, _p = self.vq(z_e)
        return indices
