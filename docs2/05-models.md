# 05 · Model Specifications

Every neural model in the project, with layer structure, tensor shapes, and config
defaults. Two families: the **mesh autoencoders** (v1/v2/v3, discrete pipeline) and
the **decoupled models** (Sky Encoder + Earth Lens — summarized here, full spec in
[09](09-decoupled-engine.md)).

All modules require torch (`import kalachakra.models.<x>` directly; the package
`__init__` does not import torch).

---

## 1. Building blocks

### 1.1 `spherical_conv.py` — geodesic convolution

`build_knn(grid, k) → (N, k)` precomputes each node's `k` nearest neighbors (incl.
self) by great‑circle proximity (pure numpy, once).

`GeodesicConv(in_ch, out_ch, neighbors)` — isotropic message passing on a fixed mesh:

```
input/output (batch, N, C)
gather neighbor features (batch, N, k, C_in) → mean over k → agg (batch, N, C_in)
out = self_lin(x) + neigh_lin(agg)          # self weight + shared neighbor weight
```

Rotation‑equivariant (neighborhoods are angular). `neighbors` is a registered buffer
so it moves with the module.

`SphericalPool(conv, stride)` — coarsen the mesh: `relu(conv(x))` then strided node
selection (nodes are Fibonacci‑ordered, so striding preserves global coverage).

### 1.2 `fno.py` — temporal Fourier Neural Operator

`SpectralConv1d(in_ch, out_ch, modes)` — the core FNO operator over **time**:

```
x (…, C_in, T) → rfft along time → keep lowest `modes` frequencies
multiply by a learned complex weight  W (in, out, modes, 2)   # real/imag stored real
→ irfft back to (…, C_out, T)
```

A continuous integral operator: discretization‑invariant, so 24 s micro‑rotation and
millennial precession are learned in the same spectral basis with no phase lag.

`FourierBlock1d` — FNO block: spectral path + a pointwise (1×1) residual path +
activation.

---

## 2. `autoencoder.py` — Spherical Autoencoder + STFNO (v1)

The blueprint Phase‑3 core.

```python
@dataclass AutoencoderConfig:
    n_nodes = 122880   in_features = 50   hidden = 128
    latent = 64        fourier_modes = 32 knn = 7   n_blocks = 3
```

`STBlock` — one spatio‑temporal block: **spatial geodesic conv then temporal FNO**
(`_apply_spatial`, `_apply_temporal`).

`SphericalAutoencoder`:

```
encode(e (B,T,N,50)):  lift → n_blocks × STBlock (spatial+temporal mix) → to 64-D latent
decode(z):             mirror network reconstructs E
forward(e) → (recon, z)
```

The 64‑D bottleneck `z(t,s)` is the continuous latent manifold (`LATENT_DIM`). Each
code summarizes a temporal window × spatial neighborhood, forcing geometric
invariants over memorized samples. The decoder provides the unsupervised
reconstruction signal.

---

## 3. `autoencoder_v2.py` — node‑chunked (MPS / large mesh)

`SphericalAutoencoderV2(SphericalAutoencoder)` and `STBlockV2(STBlock)` apply the
spatial and temporal ops in **node slices** (`_apply_spatial_chunked`,
`_apply_temporal_chunked`, `_run_blocks`) so the 122,880‑node mesh fits in unified
memory. **Same parameters** as v1 — only the execution is tiled. Use for full‑mesh
training on Apple MPS.

---

## 4. `autoencoder_v3.py` — discrete VQ bottleneck (self‑contained)

The current production model (`scripts/train_v3.py`, the Great Indexer's checkpoint
format). Fully self‑contained — its own conv/FNO/ST‑block/VQ; does **not** import
v1/v2.

```python
@dataclass VQAutoencoderV3Config:
    n_nodes = 122880   in_features = 50   hidden = 128   latent = 64
    fourier_modes = 32 knn = 7   n_blocks = 3
    codebook_size = 4096      commitment_beta = 0.25
    ema_decay = 0.99          ema_eps = 1e-5      restart_after = 10
    node_chunk = 8192         vq_chunk = 131072   grad_checkpoint = False
```

**Own building blocks:** `GeodesicConvV3` (same isotropic self+mean‑neighbor kernel),
`SpectralConv1dV3` (fp32 FFT, real‑stored complex weights; a length‑1 short‑circuit
avoids an `irfft` UserWarning), `FourierBlockV3`, `STBlockV3` (node‑chunked spatial
conv → temporal FNO, both with residuals). `build_knn(xyz, k)` here takes raw `xyz`.

**`VectorQuantizer(dim=64, codebook_size=4096, beta, decay, eps, restart_after)`** —
the discrete bottleneck:

- **Cosine / L2‑normalized quantization.** Both `z_e` and the codebook are unit‑norm;
  assignment is nearest code on the unit sphere (`_assign`).
- **EMA codebook.** The codebook is a **buffer** (no gradient tug‑of‑war): `cluster_size`
  and `embed_avg` are EMA‑updated (`_ema_update`), then the codebook is renormalized.
  Laplace smoothing (`ema_eps`) stabilizes rarely‑used codes.
- **Commitment‑only loss.** Since EMA handles the codebook term, the VQ loss is just
  `β · ‖z_e − sg[e_q]‖²` (straight‑through estimator on the forward pass).
- **Dead‑code restart.** Codes unused for `restart_after` steps are re‑seeded from
  live batch vectors (`_replace_dead_codes`), keeping perplexity high.
- Exposes `last_perplexity = exp(entropy of the assignment histogram)` and
  `lookup(indices)` (decode tokens → unit codebook vectors).

**`VQAutoencoderV3`** wiring:

```
lift: Linear(50 → hidden)
enc:  n_blocks × STBlockV3
to_latent: Linear(hidden → 64)     → z_e
vq:   VectorQuantizer               → z_q, indices, vq_loss   (cosine, EMA)
from_latent: Linear(64 → hidden)
dec:  n_blocks × STBlockV3
project out: Linear(hidden → 50)
forward(e) → (recon, z, indices, vq_loss)
tokenize(e) → per-node code indices (the 4096-archetype token stream)
```

Checkpoint payload (`train_v3.save`): `format="kalachakra-vqmodel-v3"`,
`projection_version`, `config`, `neighbors`, `grid_xyz`, `state_dict`,
`optimizer/scheduler/step`. This is what the Great Indexer's `model_io.load_model_and_grid`
consumes.

---

## 5. `rvq.py` — hierarchical residual VQ (tokens)

A separate two‑level quantizer for the tokenized index (used by the *quantized*
autoencoders below, distinct from v3's inline VQ):

```python
@dataclass RVQConfig:
    dim = 64   n_macro = 64   n_micro = 64   → n_leaf = 4096
    beta = 0.25   decay = 0.99   eps = 1e-5   dead_threshold = 1e-3
```

`HierarchicalResidualVQ` — macro code first, then a **macro‑conditioned** micro code
on the residual:

```
z (…, 64) → macro_idx (nearest of 64) → residual → micro_idx (nearest of 64, per-macro)
leaf = macro_idx · 64 + micro_idx        # 4096 leaves
```

EMA codebooks (`_ema_update`), dead‑code re‑seeding (`_replace_dead_codes`),
`lookup(leaf_idx)`. The 64 macro × 64 micro = 4096 archetype tree is the token space
the Rarity Index and news radar operate on ([07](07-inference-and-analysis.md)).

---

## 6. `quantized_autoencoder.py` / `_v2.py`

`QuantizedSphericalAutoencoder` — a `SphericalAutoencoder` whose latent is discretized
by a `HierarchicalResidualVQ`:

```
encode_quantize(e) → (z_q, leaf indices, vq_loss)
tokenize(e) → macro/micro/leaf token stream
forward(e) → (recon, z_q, tokens, vq_loss)
```

`QuantizedSphericalAutoencoderV2(...)` — the node‑chunked (MPS/large‑mesh) variant.
These are the pre‑v3 route to discrete tokens; v3 folds VQ inline. Tested in
`tests/test_quantized_ae.py`, `tests/test_rvq.py`.

---

## 7. Decoupled models (summary)

The continuous architecture (`decoupled_engine/`), full spec in
[09](09-decoupled-engine.md):

- **`SkyEncoder`** (`sky_encoder.py`) — a pre‑norm Transformer over the ten bodies as
  tokens (≥8 heads, learnable body identity + CLS summary token) → residual MLP →
  **512‑D L2‑normalized tension vector**. Exposes per‑head attention and
  `planetary_attribution`.
  Input `(B, 10, 5)` → output `(B, 512)`.
- **`EarthLensDecoder`** (`earth_lens.py`) — a coordinate implicit field (random
  Fourier features on the unit‑sphere lift + residual MLP with Gaussian/sine
  activations) mapping `(tension (M,512), latlon (M,P,2)) → OKLab (M,P,3)`. Evaluates
  any point set — pinpoint, bbox, or global grid — with no fixed discretization.

---

## 8. Model selection cheat‑sheet

| Model | Bottleneck | Spatial | Use |
|---|---|---|---|
| `SphericalAutoencoder` (v1) | continuous 64‑D | full mesh, unchunked | small meshes, reference |
| `SphericalAutoencoderV2` | continuous 64‑D | node‑chunked | full mesh on MPS |
| `VQAutoencoderV3` | **discrete 4096** (cosine EMA VQ) | node‑chunked | production tokens, indexer |
| `QuantizedSphericalAutoencoder(V2)` | discrete 4096 (RVQ tree) | (chunked) | tokenized index route |
| `SkyEncoder` + `EarthLensDecoder` | 512‑D tension vector | **continuous/query‑any‑point** | live energy field |
