"""version10 model — the Topocentric Micro Self-Attention engine.

    input   x : [N, 13, 3]         13 tokens (11 bodies + ASC + MC), each (North,East,Zenith)
    embed   t = x W_in^T + b_in + E_body[b]           -> [N, 13, D]   (+ learned body identity)
    block(s):
        Q,K,V = t Wq^T+bq , t Wk^T+bk , t Wv^T+bv     single head, d_k = D
        A     = softmax(temp * norm(Q).norm(K)^T)     -> [N, 13, 13]   (v10.1 bounded cosine attn:
        t     = t + A V                               residual         Q,K L2-normalised, temp<=30)
        t     = t + W2 tanh(W1 t + b1) + b2           per-token FFN, residual
    pool    w = softmax(temp_pool * norm(t).norm(q_pool)) over the 13 tokens
            p = sum_b w_b t_b                         -> [N, D]        (energy read-out)
    head    z = Wo2 tanh(Wo1 p + bo1) + bo2           -> [N, 2]  raw logits (Cartesian a,b axes)
    chroma  (a,b) = cmax * z / sqrt(1+|z|^2)          v10.1 PURE-CARTESIAN OKLab head (disk of
                                                      radius cmax) — NO hue angle, so the optimiser
                                                      cannot wind the hue (the deep-training beaded
                                                      zipper). NO luminance; a fixed neutral OKLab L
                                                      is added only at render.

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
from torch.nn import functional as F


def bound_cartesian(z: torch.Tensor, okl_cmax: float) -> torch.Tensor:
    """Pure-Cartesian OKLab (a, b) head — v10.1's fundamental cure for HUE WINDING.

    The two raw logits map DIRECTLY to the OKLab chroma axes, squashed into an OPEN DISK of
    radius ``okl_cmax``:

        (a, b) = okl_cmax * z / sqrt(1 + |z|^2)          |(a,b)| < okl_cmax

    There is **no angle variable** — no ``sin``/``cos`` of an unbounded accumulator — so the
    optimiser physically cannot 'spin' the hue through the colour wheel (which was free under the
    old polar ``H = z1`` head and produced the beaded rainbow zipper at depth). To wind now, the
    2-D output would have to genuinely oscillate, which the network's smoothness + weight decay
    resist and the isometric metric penalises. Near the origin the map is ~``cmax * z`` (linear,
    as the source PRD asks); it saturates smoothly to chroma <= cmax so colours stay in gamut and
    Euclidean distance on (a, b) stays perceptually meaningful (a disk, no corner clipping).
    A fixed neutral OKLab L is supplied only at render time (no luminance is optimised)."""
    r2 = (z ** 2).sum(dim=-1, keepdim=True)
    return okl_cmax * z / torch.sqrt(1.0 + r2)


def bound_oklch(z: torch.Tensor, okl_cmax: float) -> torch.Tensor:
    """Legacy polar OKLCH head (pre-v10.1): C = cmax*sigmoid(z0), H = z1, (a,b) = (C cosH, C sinH).
    Retained only for loading/rendering old weights; the trig hue angle is what made deep-training
    hue winding free, so live models use :func:`bound_cartesian` instead."""
    C = okl_cmax * torch.sigmoid(z[..., 0:1])
    H = z[..., 1:2]
    return torch.cat([C * torch.cos(H), C * torch.sin(H)], dim=-1)


class AttnBlock(nn.Module):
    """Single-head self-attention + per-token FFN, both residual (no LayerNorm — see module doc).

    When ``qk_norm`` is set (v10.1 default), Q and K are L2-normalised before their dot product,
    so the pre-softmax logit is a cosine similarity in ``[-1, 1]`` scaled by a learnable
    temperature clamped to ``[~0, temp_max]``. That bounds the logit for all time — deep training
    can inflate the weights without the softmax ever saturating into a hard switch. The legacy
    unbounded ``(1/sqrt(d)) * tau`` path is kept for loading pre-v10.1 checkpoints."""

    def __init__(self, d_model: int, d_ff: int, qk_norm: bool = True,
                 temp_init: float = 10.0, temp_max: float = 30.0):
        super().__init__()
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.qk_norm = qk_norm
        self.temp_max = temp_max
        self.base_scale = 1.0 / math.sqrt(d_model)
        #: learnable temperature. With qk_norm it is the cosine logit scale (init 10, clamped to
        #: temp_max); without it, the legacy 1/sqrt(d) softmax temperature (init 1.0).
        self.temp = nn.Parameter(torch.tensor(temp_init if qk_norm else 1.0))

    def eff_temp(self) -> torch.Tensor:
        """The effective (clamped) temperature actually multiplied into the scores."""
        return self.temp.clamp(1e-2, self.temp_max) if self.qk_norm else self.temp

    def forward(self, t: torch.Tensor, vis: torch.Tensor) -> torch.Tensor:
        # vis[...,j] is the horizon-visibility bias of key body j (vis_bias * zenith_j), added
        # to every query's score for j: above-horizon bodies are attended, below-horizon ones
        # suppressed — the "conjunction overhead spikes / underfoot zeroes" prior, observer-
        # dependent by construction. The learned Q·K content term modulates on top.
        q, k, v = self.q(t), self.k(t), self.v(t)          # [N,13,D]
        if self.qk_norm:                                   # bounded cosine attention (v10.1)
            qn, kn = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
            scores = torch.matmul(qn, kn.transpose(-1, -2)) * self.eff_temp()
        else:                                              # legacy unbounded dot-product
            scores = torch.matmul(q, k.transpose(-1, -2)) * (self.base_scale * self.temp)
        scores = scores + vis.unsqueeze(1)                 # broadcast bias over the query axis
        a = torch.softmax(scores, dim=-1)
        t = t + torch.matmul(a, v)                         # residual attention
        t = t + self.ff2(torch.tanh(self.ff1(t)))          # residual FFN
        return t


class TopoAttention(nn.Module):
    """``[N,13,3]`` topocentric tokens -> ``[N,2]`` OKLCH chroma (a,b)."""

    def __init__(self, n_bodies: int = 13, token_dim: int = 3, d_model: int = 32,
                 d_ff: int = 64, d_head: int = 32, n_blocks: int = 2, vis_bias: float = 3.0,
                 n_anchors: int = 2, okl_l: float = 0.5, okl_cmax: float = 0.4,
                 qk_norm: bool = True, attn_temp_init: float = 10.0, attn_temp_max: float = 30.0):
        super().__init__()
        self.cfg = {"n_bodies": n_bodies, "token_dim": token_dim, "d_model": d_model,
                    "d_ff": d_ff, "d_head": d_head, "n_blocks": n_blocks, "vis_bias": vis_bias,
                    "n_anchors": n_anchors, "okl_l": okl_l, "okl_cmax": okl_cmax,
                    "qk_norm": qk_norm, "attn_temp_init": attn_temp_init,
                    "attn_temp_max": attn_temp_max}
        self.vis_bias = vis_bias
        #: the last n_anchors tokens (ASC, MC) are structural coordinate axes, NOT physical
        #: bodies. They are EXEMPT from the horizon-visibility prior (which would zero out the
        #: Ascendant, since it sits on the horizon at zenith~0): they get the full vis_bias
        #: regardless of altitude, so they compete equally with each other and are not suppressed.
        self.n_anchors = n_anchors
        self.qk_norm = qk_norm
        self.temp_max = attn_temp_max
        self.embed = nn.Linear(token_dim, d_model)
        self.body_emb = nn.Parameter(torch.randn(n_bodies, d_model) * 0.02)
        self.blocks = nn.ModuleList([AttnBlock(d_model, d_ff, qk_norm, attn_temp_init,
                                               attn_temp_max) for _ in range(n_blocks)])
        self.q_pool = nn.Parameter(torch.randn(d_model) * 0.02)
        #: learnable pool-attention temperature (cosine logit scale when qk_norm, else 1/sqrt(d)).
        self.temp_pool = nn.Parameter(torch.tensor(attn_temp_init if qk_norm else 1.0))
        self.head1 = nn.Linear(d_model, d_head)
        self.head2 = nn.Linear(d_head, 2)                  # pure chroma: a*, b* (no L*)
        self.base_pool_scale = 1.0 / math.sqrt(d_model)
        self._reset()

    def eff_temp_pool(self) -> torch.Tensor:
        return self.temp_pool.clamp(1e-2, self.temp_max) if self.qk_norm else self.temp_pool

    def _reset(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def _pool_weights(self, t: torch.Tensor, vis: torch.Tensor) -> torch.Tensor:
        if self.qk_norm:                                   # bounded cosine pool (v10.1)
            tn = F.normalize(t, dim=-1)                    # [N,NB,D]
            qn = F.normalize(self.q_pool, dim=0)           # [D]
            scores = torch.matmul(tn, qn) * self.eff_temp_pool() + vis
        else:                                              # legacy unbounded dot-product
            scores = torch.matmul(t, self.q_pool) * (self.base_pool_scale * self.temp_pool) + vis
        return torch.softmax(scores, dim=1)                # [N,NB] energy contribution per token

    def forward(self, x: torch.Tensor, return_pool: bool = False):
        """``x``: ``[N,13,3]`` (or ``[N,39]``, auto-reshaped) -> ``[N,2]`` OKLab (a,b)."""
        if x.dim() == 2:
            x = x.view(x.shape[0], self.cfg["n_bodies"], self.cfg["token_dim"])
        zen = x[..., 2]                                    # zenith component per token
        if self.n_anchors > 0:                             # ASC/MC: always fully visible (z:=1)
            zen = torch.cat([zen[..., :-self.n_anchors],
                             torch.ones_like(zen[..., -self.n_anchors:])], dim=-1)
        vis = self.vis_bias * zen                          # [N,NB] visibility bias (bodies gated)
        t = self.embed(x) + self.body_emb                  # [N,11,D]
        for blk in self.blocks:
            t = blk(t, vis)
        w = self._pool_weights(t, vis)                     # [N,11]
        pooled = torch.einsum("nb,nbd->nd", w, t)          # [N,D]
        z = self.head2(torch.tanh(self.head1(pooled)))     # [N,2] raw logits (Cartesian a,b axes)
        ab = bound_cartesian(z, self.cfg["okl_cmax"])      # [N,2] OKLab (a,b) — no hue angle
        return (ab, w) if return_pool else ab

    def export_weights(self) -> dict:
        def W(m):  # [out][in]
            return m.weight.detach().cpu().tolist()

        def b(m):
            return m.bias.detach().cpu().tolist()

        c = self.cfg
        # "tau" / "tau_pool" carry the EFFECTIVE (clamped) temperature so the JS/GLSL ports just
        # multiply the (normalised, when qk_norm) dot product by it — no clamp logic downstream.
        blocks = [{
            "Wq": W(blk.q), "bq": b(blk.q), "Wk": W(blk.k), "bk": b(blk.k),
            "Wv": W(blk.v), "bv": b(blk.v),
            "W1": W(blk.ff1), "b1": b(blk.ff1), "W2": W(blk.ff2), "b2": b(blk.ff2),
            "tau": float(blk.eff_temp().detach()),
        } for blk in self.blocks]
        return {
            "arch": "v10_topo_attention", "output_activation": "v10_cartesian", "out_features": 2,
            "n_bodies": c["n_bodies"], "token_dim": c["token_dim"], "d_model": c["d_model"],
            "d_ff": c["d_ff"], "d_head": c["d_head"], "n_blocks": c["n_blocks"],
            "vis_bias": c["vis_bias"], "n_anchors": c["n_anchors"],
            "qk_norm": bool(c["qk_norm"]),
            "okl_l": c["okl_l"], "okl_cmax": c["okl_cmax"],
            "W_in": W(self.embed), "b_in": b(self.embed),
            "E_body": self.body_emb.detach().cpu().tolist(),
            "blocks": blocks, "q_pool": self.q_pool.detach().cpu().tolist(),
            "tau_pool": float(self.eff_temp_pool().detach()),
            "Wo1": W(self.head1), "bo1": b(self.head1),
            "Wo2": W(self.head2), "bo2": b(self.head2),
        }


def build_model(**kw) -> TopoAttention:
    return TopoAttention(**kw)
