"""version9 model — the Topocentric Micro Self-Attention engine.

    input   x : [N, 11, 3]         11 body tokens, each (North, East, Zenith), per observer
    embed   t = x W_in^T + b_in + E_body[b]           -> [N, 11, D]   (+ learned body identity)
    block(s):
        Q,K,V = t Wq^T+bq , t Wk^T+bk , t Wv^T+bv     single head, d_k = D
        A     = softmax(Q K^T / sqrt(D))              -> [N, 11, 11]   (topocentric relations)
        t     = t + A V                               residual
        t     = t + W2 tanh(W1 t + b1) + b2           per-token FFN, residual
    pool    w = softmax((t . q_pool)/sqrt(D)) over the 11 tokens
            p = sum_b w_b t_b                         -> [N, D]        (energy read-out)
    head    z = Wo2 tanh(Wo1 p + bo1) + bo2           -> [N, 2]
    chroma  a*,b* = ab*tanh(z0,z1)                    pure 2-D CIE a*b* energy (NO luminance;
                                                      a fixed neutral L* is added only at render)

Why attention (not v8's fixed chords)? A learned bilinear form Q K^T = t_i^T (Wq^T Wk) t_j is
**not** rotation-invariant unless Wq^T Wk ~ I, so — unlike a plain dot product — it varies with
the observer's horizon. The softmax then sharpens near alignments/horizon crossings, giving the
localized "patches" that a smooth cosine input cannot. Body-identity embeddings make the Sun and
Pluto distinguishable (a bare attention over 3-vectors would be permutation-symmetric).

Every op here is a plain matmul / tanh / softmax so it transcribes verbatim to attn9.js and the
GLSL vertex shader (shader9.js); the numpy re-run in the tests is the JS/GLSL parity contract.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def bound_ab(z: torch.Tensor, lab_ab: float) -> torch.Tensor:
    """Pure-chroma head: a*,b* = ab*tanh(z0,z1). No luminance — the field is a 2-D CIE a*b*
    energy signature; a fixed neutral L* is supplied only at render time."""
    a = lab_ab * torch.tanh(z[..., 0:1])
    b = lab_ab * torch.tanh(z[..., 1:2])
    return torch.cat([a, b], dim=-1)


class AttnBlock(nn.Module):
    """Single-head self-attention + per-token FFN, both residual (no LayerNorm — see module doc)."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.base_scale = 1.0 / math.sqrt(d_model)
        #: learnable temperature — lets the softmax SHARPEN (spike on aligned/visible bodies)
        #: instead of collapsing to a uniform mean-pool. Exported as one scalar per block.
        self.tau = nn.Parameter(torch.tensor(1.0))

    def forward(self, t: torch.Tensor, vis: torch.Tensor) -> torch.Tensor:
        # vis[...,j] is the horizon-visibility bias of key body j (vis_bias * zenith_j), added
        # to every query's score for j: above-horizon bodies are attended, below-horizon ones
        # suppressed — the "conjunction overhead spikes / underfoot zeroes" prior, observer-
        # dependent by construction. The learned Q·K content term modulates on top.
        q, k, v = self.q(t), self.k(t), self.v(t)          # [N,11,D]
        scores = torch.matmul(q, k.transpose(-1, -2)) * (self.base_scale * self.tau)  # [N,11,11]
        scores = scores + vis.unsqueeze(1)                 # broadcast bias over the query axis
        a = torch.softmax(scores, dim=-1)
        t = t + torch.matmul(a, v)                         # residual attention
        t = t + self.ff2(torch.tanh(self.ff1(t)))          # residual FFN
        return t


class TopoAttention(nn.Module):
    """``[N,11,3]`` topocentric tokens -> ``[N,3]`` gamut-bounded CIE L*a*b*."""

    def __init__(self, n_bodies: int = 11, token_dim: int = 3, d_model: int = 32,
                 d_ff: int = 64, d_head: int = 32, n_blocks: int = 2, vis_bias: float = 3.0,
                 lab_l: float = 50.0, lab_ab: float = 80.0):
        super().__init__()
        self.cfg = {"n_bodies": n_bodies, "token_dim": token_dim, "d_model": d_model,
                    "d_ff": d_ff, "d_head": d_head, "n_blocks": n_blocks, "vis_bias": vis_bias,
                    "lab_l": lab_l, "lab_ab": lab_ab}
        self.vis_bias = vis_bias
        self.embed = nn.Linear(token_dim, d_model)
        self.body_emb = nn.Parameter(torch.randn(n_bodies, d_model) * 0.02)
        self.blocks = nn.ModuleList([AttnBlock(d_model, d_ff) for _ in range(n_blocks)])
        self.q_pool = nn.Parameter(torch.randn(d_model) * 0.02)
        self.tau_pool = nn.Parameter(torch.tensor(1.0))    # learnable pool-attention temperature
        self.head1 = nn.Linear(d_model, d_head)
        self.head2 = nn.Linear(d_head, 2)                  # pure chroma: a*, b* (no L*)
        self.base_pool_scale = 1.0 / math.sqrt(d_model)
        self._reset()

    def _reset(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def _pool_weights(self, t: torch.Tensor, vis: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(t, self.q_pool) * (self.base_pool_scale * self.tau_pool) + vis
        return torch.softmax(scores, dim=1)                # [N,11] energy contribution per body

    def forward(self, x: torch.Tensor, return_pool: bool = False):
        """``x``: ``[N,11,3]`` (or ``[N,33]``, auto-reshaped) -> ``[N,3]`` L*a*b*."""
        if x.dim() == 2:
            x = x.view(x.shape[0], self.cfg["n_bodies"], self.cfg["token_dim"])
        vis = self.vis_bias * x[..., 2]                    # [N,11] horizon-visibility bias
        t = self.embed(x) + self.body_emb                  # [N,11,D]
        for blk in self.blocks:
            t = blk(t, vis)
        w = self._pool_weights(t, vis)                     # [N,11]
        pooled = torch.einsum("nb,nbd->nd", w, t)          # [N,D]
        z = self.head2(torch.tanh(self.head1(pooled)))     # [N,2]
        ab = bound_ab(z, self.cfg["lab_ab"])               # [N,2] pure a*,b* chroma
        return (ab, w) if return_pool else ab

    def export_weights(self) -> dict:
        def W(m):  # [out][in]
            return m.weight.detach().cpu().tolist()

        def b(m):
            return m.bias.detach().cpu().tolist()

        c = self.cfg
        blocks = [{
            "Wq": W(blk.q), "bq": b(blk.q), "Wk": W(blk.k), "bk": b(blk.k),
            "Wv": W(blk.v), "bv": b(blk.v),
            "W1": W(blk.ff1), "b1": b(blk.ff1), "W2": W(blk.ff2), "b2": b(blk.ff2),
            "tau": float(blk.tau.detach()),
        } for blk in self.blocks]
        return {
            "arch": "v9_topo_attention", "output_activation": "v9_chroma", "out_features": 2,
            "n_bodies": c["n_bodies"], "token_dim": c["token_dim"], "d_model": c["d_model"],
            "d_ff": c["d_ff"], "d_head": c["d_head"], "n_blocks": c["n_blocks"],
            "vis_bias": c["vis_bias"],
            "lab_l": c["lab_l"], "lab_ab": c["lab_ab"],
            "W_in": W(self.embed), "b_in": b(self.embed),
            "E_body": self.body_emb.detach().cpu().tolist(),
            "blocks": blocks, "q_pool": self.q_pool.detach().cpu().tolist(),
            "tau_pool": float(self.tau_pool.detach()),
            "Wo1": W(self.head1), "bo1": b(self.head1),
            "Wo2": W(self.head2), "bo2": b(self.head2),
        }


def build_model(**kw) -> TopoAttention:
    return TopoAttention(**kw)
