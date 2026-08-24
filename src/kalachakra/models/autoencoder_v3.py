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

The quantizer uses the modern stable formulation — L2-normalized (cosine)
nearest-code lookup, an EMA-updated codebook (a buffer, no gradient tug-of-war),
a commitment-only loss, and dead-code restart — and returns the straight-through
z_q (gradients flow to the encoder), the discrete token indices, the VQ loss, and
the batch codebook perplexity. The codebook and its EMA statistics are registered
buffers, so they are saved in the model's state_dict.

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
    commitment_beta: float = 0.25  # beta on the commitment loss
    ema_decay: float = 0.99        # EMA decay for the codebook cluster centers
    ema_eps: float = 1e-5          # Laplace smoothing for cluster sizes
    restart_after: int = 10        # restart a code unused for this many steps
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
    """Stable VQ: L2-normalized (cosine) lookup + EMA codebook + dead-code restart.

    Modern stability best-practices (ViT-VQGAN style), which cure the classic
    gradient-VQ instability (exploding VQ loss / collapsing perplexity):

    * **Cosine / L2-normalized quantization.** ``z_e`` and the codebook are unit-
      normalized before the nearest-neighbour lookup, so on the unit sphere
      ``argmin ||z-e||^2 == argmax cosine``; all distances are bounded in [0, 4],
      which removes the magnitude blow-up.
    * **EMA codebook.** The codebook is a **buffer** (no gradient tug-of-war with
      the encoder): its clusters are moving-averaged toward the assigned encoder
      outputs (decay 0.99, Laplace-smoothed), then re-normalized to unit norm.
    * **Commitment-only loss.** ``L_VQ = beta * ||z_norm - sg[z_q]||^2`` — the
      codebook term is gone (EMA handles it). Scale further with ``lambda_vq`` in
      the trainer.
    * **Dead-code restart.** A code unused for ``restart_after`` consecutive steps
      is re-seeded from a random normalized latent in the current batch, keeping
      perplexity (codebook utilization) high.

    The codebook is a buffer, so it (and the EMA stats) are saved in ``state_dict``.
    The lookup is tiled over the token axis and runs in float32, so it is safe at
    the full mesh (~10^7 tokens/step).
    """

    def __init__(self, dim: int = 64, codebook_size: int = 4096,
                 beta: float = 0.25, decay: float = 0.99, eps: float = 1e-5,
                 restart_after: int = 10, vq_chunk: int = 131_072):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.beta = beta
        self.decay = decay
        self.eps = eps
        self.restart_after = restart_after
        self.vq_chunk = vq_chunk
        embed = F.normalize(torch.randn(codebook_size, dim), dim=1)
        self.register_buffer("codebook", embed)                 # (K, D) unit norm
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embed_avg", embed.clone())        # EMA of assigned z
        self.register_buffer("unused_steps", torch.zeros(codebook_size))

    @torch.no_grad()
    def _assign(self, z_norm: torch.Tensor, e_norm: torch.Tensor) -> torch.Tensor:
        """Nearest code per row by cosine (== normalized Euclidean), tiled."""
        m = z_norm.shape[0]
        idx = torch.empty(m, dtype=torch.long, device=z_norm.device)
        for s in range(0, m, self.vq_chunk):
            sim = z_norm[s:s + self.vq_chunk] @ e_norm.t()      # (c, K) cosine
            idx[s:s + self.vq_chunk] = sim.argmax(dim=1)
        return idx

    def forward(self, z_e: torch.Tensor):
        """z_e: (..., dim). Returns (z_q_ste, indices, vq_loss, perplexity)."""
        with torch.autocast(device_type=z_e.device.type, enabled=False):
            ze = z_e.float()
            lead = ze.shape[:-1]
            flat = ze.reshape(-1, self.dim)                     # (M, D)
            z_norm = F.normalize(flat, dim=1)                   # unit sphere
            e_norm = F.normalize(self.codebook, dim=1)          # unit sphere

            idx = self._assign(z_norm, e_norm)                  # (M,)
            z_q = F.embedding(idx, e_norm)                      # unit-norm code

            # commitment-only loss (codebook is EMA); beta-weighted.
            vq_loss = self.beta * F.mse_loss(z_norm, z_q.detach())

            # straight-through estimator: forward is z_q, gradient flows to z_e.
            z_q_ste = z_norm + (z_q - z_norm).detach()

            # batch codebook perplexity = exp(entropy of the assignment histogram).
            counts = torch.bincount(idx, minlength=self.codebook_size).float()
            probs = counts / counts.sum().clamp_min(1.0)
            perplexity = torch.exp(-(probs * (probs + 1e-10).log()).sum())

            if self.training:
                self._ema_update(z_norm, idx, counts)

        z_q_ste = z_q_ste.reshape(*lead, self.dim).to(z_e.dtype)
        indices = idx.reshape(*lead)
        return z_q_ste, indices, vq_loss.to(z_e.dtype), perplexity.to(z_e.dtype)

    @torch.no_grad()
    def _ema_update(self, z_norm: torch.Tensor, idx: torch.Tensor,
                    counts: torch.Tensor) -> None:
        """Move cluster centers toward assigned latents (EMA), then restart dead
        codes. Runs once per step (the VQ sits outside the checkpointed blocks)."""
        # EMA of cluster sizes and of the sum of assigned (normalized) latents.
        self.cluster_size.mul_(self.decay).add_(counts, alpha=1.0 - self.decay)
        embed_sum = torch.zeros_like(self.embed_avg)
        embed_sum.index_add_(0, idx, z_norm)
        self.embed_avg.mul_(self.decay).add_(embed_sum, alpha=1.0 - self.decay)
        # Laplace-smoothed normalized cluster centers, kept on the unit sphere.
        n = self.cluster_size.sum()
        smoothed = ((self.cluster_size + self.eps)
                    / (n + self.codebook_size * self.eps) * n)
        self.codebook.copy_(F.normalize(self.embed_avg / smoothed.unsqueeze(1), dim=1))

        # Dead-code restart: reseed codes unused for `restart_after` steps from a
        # random slice of this batch's normalized latents.
        used = counts > 0
        self.unused_steps = torch.where(
            used, torch.zeros_like(self.unused_steps), self.unused_steps + 1.0)
        dead = self.unused_steps >= self.restart_after
        n_dead = int(dead.sum())
        if n_dead > 0 and z_norm.shape[0] > 0:
            pick = torch.randint(0, z_norm.shape[0], (n_dead,), device=z_norm.device)
            samples = z_norm[pick]
            self.codebook[dead] = samples
            self.embed_avg[dead] = samples
            self.cluster_size[dead] = 1.0
            self.unused_steps[dead] = 0.0

    @torch.no_grad()
    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        """Unit-norm codebook vectors for token indices (decode from tokens)."""
        return F.embedding(indices, F.normalize(self.codebook, dim=1))


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

        self.vq = VectorQuantizer(
            cfg.latent, cfg.codebook_size, beta=cfg.commitment_beta,
            decay=cfg.ema_decay, eps=cfg.ema_eps, restart_after=cfg.restart_after,
            vq_chunk=cfg.vq_chunk)

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
