"""
Hierarchical Residual Vector Quantization (blueprint Phase 2, §2).

Turns the continuous 64-d latent z(t, s) into a discrete, hierarchical geometric
token without human categories. Two cascading levels:

* **Macro** codebook: ``n_macro`` (=64) basis vectors capturing broad, low-
  frequency structural symmetry. The latent is snapped to its nearest macro code.
* **Micro** codebook: a *separate* ``n_micro`` (=64) residual codebook **per macro
  cluster** — ``n_macro * n_micro = 4096`` distinct leaf tokens. The macro
  residual ``z - q_macro`` is snapped within the codebook conditioned on the macro
  index.

Each latent thus maps to a composite token ``(macro_id, micro_id)`` with
``leaf_id = macro_id * n_micro + micro_id`` in ``[0, 4096)``.

Training details (all standard VQ-VAE machinery):
- **Straight-through estimator**: gradients bypass the non-differentiable argmin,
  copied from the quantized output back to the encoder.
- **Commitment loss** (weight ``beta`` = 0.25) keeps encoder outputs near their
  chosen codes; a codebook-loss value is also reported for logging.
- **EMA codebook updates**: codes track the mean of the vectors assigned to them.
- **Dead-code replacement**: codes whose usage falls below a threshold are
  re-seeded from active latent vectors, preventing codebook collapse.

Requires PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RVQConfig:
    dim: int = 64
    n_macro: int = 64
    n_micro: int = 64
    beta: float = 0.25          # commitment weight
    decay: float = 0.99         # EMA decay
    eps: float = 1e-5           # Laplace smoothing for EMA normalization
    dead_threshold: float = 1e-3  # min usage fraction before a code is re-seeded

    @property
    def n_leaf(self) -> int:
        return self.n_macro * self.n_micro


def _nearest(x: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """Indices of the nearest codebook row for each row of x (squared L2)."""
    # x: (N, D), codebook: (K, D) -> (N,)
    d = (x.pow(2).sum(1, keepdim=True)
         - 2 * x @ codebook.t()
         + codebook.pow(2).sum(1))
    return d.argmin(1)


class HierarchicalResidualVQ(nn.Module):
    """Two-level, macro-conditioned residual vector quantizer."""

    def __init__(self, cfg: RVQConfig | None = None):
        super().__init__()
        self.cfg = cfg or RVQConfig()
        c = self.cfg

        macro = torch.randn(c.n_macro, c.dim)
        micro = torch.randn(c.n_macro, c.n_micro, c.dim)
        # Codebooks are buffers, not Parameters: they are updated by EMA, not SGD.
        self.register_buffer("macro_codebook", macro)
        self.register_buffer("micro_codebook", micro)
        # EMA state.
        self.register_buffer("macro_count", torch.zeros(c.n_macro))
        self.register_buffer("macro_sum", macro.clone())
        self.register_buffer("micro_count", torch.zeros(c.n_macro, c.n_micro))
        self.register_buffer("micro_sum", micro.clone())

    # -- forward ----------------------------------------------------------
    def forward(self, z: torch.Tensor):
        """Quantize ``z`` of shape ``(..., dim)``.

        Returns ``(quantized, info)`` where ``quantized`` has the same shape as
        ``z`` (straight-through), and ``info`` holds macro/micro/leaf indices
        (batch-shaped), the VQ loss, and per-part loss components.
        """
        c = self.cfg
        lead = z.shape[:-1]
        in_dtype = z.dtype
        # The codebooks/EMA buffers are float32; run all quantizer math in float32
        # with autocast disabled (bf16 breaks index-assignment into the codebooks
        # and mixes dtypes in the straight-through), then cast back to the caller.
        with torch.autocast(device_type=z.device.type, enabled=False):
            flat = z.reshape(-1, c.dim).float()

            # --- macro level ---
            macro_idx = _nearest(flat, self.macro_codebook)          # (N,)
            q_macro = self.macro_codebook[macro_idx]                  # (N, D)
            residual = flat - q_macro

            # --- micro level (codebook conditioned on macro index) ---
            cond = self.micro_codebook[macro_idx]                    # (N, n_micro, D)
            d = (residual.unsqueeze(1) - cond).pow(2).sum(-1)        # (N, n_micro)
            micro_idx = d.argmin(1)                                  # (N,)
            n = flat.shape[0]
            q_micro = cond[torch.arange(n, device=flat.device), micro_idx]  # (N, D)

            quantized = q_macro + q_micro

            # --- losses ---
            codebook_loss = (F.mse_loss(q_macro, flat.detach())
                             + F.mse_loss(q_micro, residual.detach()))
            commitment = (F.mse_loss(flat, q_macro.detach())
                          + F.mse_loss(residual, q_micro.detach()))
            vq_loss = c.beta * commitment

            # --- EMA + dead-code (training only) ---
            if self.training:
                self._ema_update(flat, residual, macro_idx, micro_idx)

            # --- straight-through ---
            quantized_st = flat + (quantized - flat).detach()
            quantized_st = quantized_st.reshape(*lead, c.dim).to(in_dtype)

        leaf_idx = macro_idx * c.n_micro + micro_idx
        info = {
            "macro_idx": macro_idx.reshape(lead),
            "micro_idx": micro_idx.reshape(lead),
            "leaf_idx": leaf_idx.reshape(lead),
            "vq_loss": vq_loss,
            "codebook_loss": codebook_loss.detach(),
            "commitment_loss": commitment.detach(),
        }
        return quantized_st, info

    # -- EMA + dead-code --------------------------------------------------
    @torch.no_grad()
    def _ema_update(self, flat, residual, macro_idx, micro_idx):
        c = self.cfg
        # Macro EMA.
        onehot = F.one_hot(macro_idx, c.n_macro).type_as(flat)     # (N, n_macro)
        self.macro_count.mul_(c.decay).add_(onehot.sum(0), alpha=1 - c.decay)
        self.macro_sum.mul_(c.decay).add_(onehot.t() @ flat, alpha=1 - c.decay)
        n_macro_total = self.macro_count.sum()
        macro_w = ((self.macro_count + c.eps)
                   / (n_macro_total + c.n_macro * c.eps) * n_macro_total)
        self.macro_codebook.copy_(self.macro_sum / macro_w.unsqueeze(1))

        # Micro EMA (per macro cluster).
        flat_cell = macro_idx * c.n_micro + micro_idx             # (N,)
        cell_onehot = F.one_hot(flat_cell, c.n_macro * c.n_micro).type_as(flat)
        counts = cell_onehot.sum(0).reshape(c.n_macro, c.n_micro)
        sums = (cell_onehot.t() @ residual).reshape(c.n_macro, c.n_micro, c.dim)
        self.micro_count.mul_(c.decay).add_(counts, alpha=1 - c.decay)
        self.micro_sum.mul_(c.decay).add_(sums, alpha=1 - c.decay)
        micro_w = (self.micro_count + c.eps).unsqueeze(-1)
        self.micro_codebook.copy_(self.micro_sum / micro_w)

        self._replace_dead_codes(flat, residual)

    @torch.no_grad()
    def _replace_dead_codes(self, flat, residual):
        c = self.cfg
        n = flat.shape[0]
        # Macro dead codes -> re-seed from random active latents.
        macro_prob = self.macro_count / self.macro_count.sum().clamp_min(1.0)
        dead = torch.where(macro_prob < c.dead_threshold)[0]
        if dead.numel() and n:
            pick = torch.randint(0, n, (dead.numel(),), device=flat.device)
            self.macro_codebook[dead] = flat[pick]
            self.macro_sum[dead] = flat[pick]
            self.macro_count[dead] = 1.0
        # Micro dead codes -> re-seed from random residuals.
        micro_prob = self.micro_count / self.micro_count.sum().clamp_min(1.0)
        dm, dmi = torch.where(micro_prob < c.dead_threshold / c.n_macro)
        if dm.numel() and n:
            pick = torch.randint(0, n, (dm.numel(),), device=flat.device)
            self.micro_codebook[dm, dmi] = residual[pick]
            self.micro_sum[dm, dmi] = residual[pick]
            self.micro_count[dm, dmi] = 1.0

    # -- inference helpers ------------------------------------------------
    @torch.no_grad()
    def lookup(self, leaf_idx: torch.Tensor) -> torch.Tensor:
        """Reconstruct quantized vectors from leaf ids ``(...)`` -> ``(..., dim)``."""
        c = self.cfg
        macro_idx = leaf_idx // c.n_micro
        micro_idx = leaf_idx % c.n_micro
        q_macro = self.macro_codebook[macro_idx]
        q_micro = self.micro_codebook[macro_idx, micro_idx]
        return q_macro + q_micro
