# 09 · The Decoupled Projection Engine

`decoupled_engine/` — a continuous, mesh‑free replacement for the discrete
122,880‑node / VQ‑codebook architecture. It separates **celestial geometry** (the
Sky Encoder) from **terrestrial spatial mapping** (the Earth Lens Decoder). The
energy field is queryable at any `(lat, lon)`, at infinite resolution, and rendered
live over the globe.

Files: `config, features, dataset, sky_encoder, earth_lens, color, losses, training,
inference, bundle, export, api, __main__`.

---

## 1. Philosophy

- **The sky is one object.** At an instant, the ten bodies' positions are
  location‑free. The **Sky Encoder** compresses them into a single 512‑D **global
  tension vector** `z`.
- **The ground is a query.** The **Earth Lens** is an implicit field `f(z, lat, lon)
  → OKLab colour`. Run the encoder once per instant; evaluate the lens anywhere.
- **Continuous & differential‑geometry‑grounded.** No mesh, no codebook, no static
  discretization. Every angle is `(sin, cos)`; coordinates are lifted to the unit
  sphere so there is no ±180° seam or pole singularity.

The ten bodies here are **Sun…Pluto** (swe ids 0–9 — adds Uranus/Neptune/Pluto,
drops the nodes/Ayanamsha of the original `G(t)`).

---

## 2. `config.py` — all hyperparameters

```python
N_BODIES = 10   BODY_FEATURES = 5

@dataclass SkyEncoderConfig:
    n_bodies=10  in_features=5  d_model=128  nhead=8  num_layers=4
    dim_feedforward=256  dropout=0.0  tension_dim=512
    normalize_output=True  grad_checkpoint=False

@dataclass EarthLensConfig:
    tension_dim=512  coord_dim=3           # (lat,lon) lifted to a unit 3-vector
    num_fourier=64  fourier_scale=8.0  learnable_fourier=False
    hidden=256  n_blocks=4  activation="gauss"   # "gauss" | "sine"
    gauss_sigma=0.1  sine_omega0=30.0
    out_channels=3   bound_output=True  ab_scale=0.4   # OKLab (L,a,b)

@dataclass DataConfig:          # see 04 §3
    start_jd, end_jd  temporal_len=3  stride_seconds=3600
    points_per_frame=1024  samples_per_epoch=4096  seed=0

@dataclass TrainConfig:
    lr=3e-4  weight_decay=1e-2  warmup_steps=200  max_steps=5000
    batch_size=8  grad_clip=1.0
    w_geometric=1.0  w_terrestrial=0.5  w_temporal=0.25
    geo_temperature=0.1  geodesic_eps_deg=0.5  amp=True
    device=""  out_dir="checkpoints/decoupled"  save_every=250  log_every=10  seed=0

@dataclass EngineConfig:  sky; earth; data; train
    # __post_init__ asserts earth.tension_dim == sky.tension_dim
    to_dict() / from_dict(d)
```

---

## 3. `features.py` — celestial & terrestrial encoding

**Bodies:** `BODY_SWE_IDS = (0..9)`, `BODY_NAMES = (Sun,Moon,Mercury,Venus,Mars,
Jupiter,Saturn,Uranus,Neptune,Pluto)`.

| Function | Meaning |
|---|---|
| `raw_bodies(jd)` | `(lon, lat, lon_velocity)` in rad / rad / **rad·day⁻¹** via `swe.calc_ut` with the speed flag |
| `encode_celestial(lon, lat, vel)` | `(10,5)` = `[sinλ, cosλ, sinβ, cosβ, λ̇]` (float32) |
| `celestial_features(jd)` / `_batch(jds)` | one / many instants → `(10,5)` / `(T,10,5)` |
| `decode_lonlat_np(f)` / `decode_lon(f)` | recover `(lon,lat)` (numpy / torch) |
| `sample_sphere_coords(n, rng)` | `n` random `(lat,lon)`, **area‑uniform** (`sin lat ~ U(-1,1)`) |
| `equirect_grid(w, h)` | `(h·w, 2)` equirectangular grid, north‑up |
| `latlon_to_unit_vector(latlon)` | `(…,2)`→`(…,3)` unit vector — the wrap‑free lift |
| `geodesic_neighbor(latlon, eps_rad, gen)` | a point a true great‑circle step `eps` away on a random bearing |

Velocity is kept in **rad/day** — a natural unit, no arbitrary rescaling.

`dataset.py` (`CelestialTerrestrialStream`, `move_batch`, `build_dataloader`) is
detailed in [04 §3](04-data-pipelines.md).

---

## 4. Losses (`losses.py`) — the math

Three self‑supervised terms; no labels.

### 4.1 Wave‑mechanics descriptor

`harmonic_interference_descriptor(celestial, harmonics=(1,2,3,4,6))`:

```
λ = decode_lon(celestial)                       # (M,10)
for each body pair (i<j), each harmonic k:  cos(k·(λ_i − λ_j))
→ descriptor g (M, 45·5 = 225)
```

Each feature is the standing‑wave alignment of an aspect (k=1 conjunction/opposition,
2 square, 3 trine/sextile, …). This is the geometric ground truth.

### 4.2 Geometric interference **contrastive** loss

`geometric_interference_contrastive_loss(z, celestial, temperature=0.1)` — a soft
contrastive objective aligning the latent neighbourhood to the geometric one:

```
z ← L2-normalize(z) ;  g ← L2-normalize(descriptor(celestial))
S_g = g gᵀ / τ  (diag → −∞) ;  target = softmax(S_g, dim=1)
S_z = z zᵀ / τ  (diag → −∞) ;  logQ   = log_softmax(S_z, dim=1)
loss = − Σ target · logQ   (mean over the batch)
```

**Anti‑collapse:** if all `z` are identical, `logQ` is uniform against a peaked
target → high loss ≈ `log(M−1)` (verified in tests: `loss(collapsed) > loss(aligned)`).

### 4.3 Culmination edge permission

`culmination_edge_permission(celestial, coords, gmst_rad, sigma=0.15)` — where a
sharp colour edge is physically permitted (a planet on the local meridian):

```
local_sid = point_lon + gmst           # per point
sep = angular_sep(local_sid, body_lon) # (M,P,10), wrap-safe
permission = maxₖ exp(−(sep/σ)²)        # (M,P) ∈ (0,1]
```

### 4.4 Terrestrial smoothness loss

`terrestrial_smoothness_loss(color, color_neighbor, delta_geodesic, edge_permission)`:

```
grad² = ‖color − color_neighbor‖² / delta_geodesic²     # geodesic gradient
loss  = mean( (1 − permission) · grad² )
```

Smoothness is enforced everywhere **except** at active culmination boundaries (where
Ascendant/culmination edges are physical).

### 4.5 Temporal continuity loss

`temporal_continuity_loss(color_seq (B,T,P,3))` — the **second time‑difference**:

```
d = color[t+1] − 2·color[t] + color[t−1] ;  loss = mean(d²)   (T≥3; falls back for T=2)
```

Zero for steady drift (a fast but smooth lunar transit), large for erratic spikes.

---

## 5. `sky_encoder.py` — the transformer

`SkyEncoder(SkyEncoderConfig)`:

```
embed:      Linear(5 → d_model)
tokens:     embed(x) + body_embed(1,10,d)  → LayerNorm ;  prepend CLS(1,1,d)  → (B,11,d)
layers:     num_layers × AspectAttentionLayer   (pre-norm MHA, ≥8 heads, GELU FF)
summary:    seq[:,0]  (the CLS/global summary token)
head:       ResidualMLP(d) → LayerNorm → Linear(d → 512)
output:     z = F.normalize(head(summary))       # (B,512) unit tension vector
```

- `AspectAttentionLayer` — pre‑norm block using `nn.MultiheadAttention` that can
  **return per‑head attention** (`average_attn_weights=False`).
- `forward(x, return_attention=False)` → `z (B,512)`, or `(z, attn (B,L,H,11,11))`.
- `planetary_attribution(x) → (B,10)` — the CLS token's attention onto each body,
  averaged over heads & layers, normalized to sum 1 (dynamic, table‑free attribution).
- `grad_checkpoint` wraps each layer in `torch.utils.checkpoint`; AMP‑friendly.
- A learnable **body‑identity embedding** distinguishes the ten tokens; the attention
  matrix computes all‑to‑all aspect interference with no hardcoded aspect table.

Input `(B,10,5)` → output `(B,512)`.

---

## 6. `earth_lens.py` — the implicit field

`EarthLensDecoder(EarthLensConfig)`:

```
uv        = latlon_to_unit_vector(latlon)          # (M,P,3) continuous on S²
feats     = FourierFeatures(uv)                    # [sin(2π·uv·B), cos(...)] → (M,P,2·num)
h         = act( input_proj( cat[feats, tension] ) )   # tension broadcast over P
h         = n_blocks × INRBlock(h)                 # x + W2·act(W1·act(x))
out       = head(h)                                # (M,P,3)
if bound_output:  L = ½(tanh(out₀)+1)∈[0,1] ;  a,b = 0.4·tanh(out₁,₂)   # OKLab domain
```

- `FourierFeatures(in_dim=3, num=64, scale=8, learnable=False)` — random Fourier
  features defeat the MLP spectral bias so sharp Ascendant transitions resolve.
  The unit‑sphere lift makes them seam‑free.
- Activation `gauss` (`exp(−x²/2σ²)`) or `sine` (SIREN, with `_siren_init`).
- `forward(tension, latlon)` accepts `tension (M,512)` or `(512,)` and `latlon
  (M,P,2)` or `(P,2)` (broadcast), returns `(M,P,3)` (or `(P,3)` unbatched).
- **Coordinate‑agnostic:** one point, a 5‑km bbox, or a global grid — no architectural
  change, no fixed discretization.

`color.py` — OKLab↔sRGB (Björn Ottosson): `oklab_to_linear_srgb`, `oklab_to_srgb`,
`oklab_to_srgb8` (uint8, for the WebGL byte buffer). Pure numpy.

---

## 7. Training, inference, export, serving

- **`training.py`** — `composite_step(sky, earth, batch, cfg, device)`:
  ```
  z = sky(cel.reshape(B·T,10,5))                        # (B·T,512)
  color = earth(z, coords expanded over T)              # (B·T,P,3)
  L = w_geo·contrastive(z, cel) + w_terr·smoothness(color, color_nb, ε, permission)
      + w_temp·temporal(color.reshape(B,T,P,3))
  ```
  where `color_nb = earth(z, geodesic_neighbor(coords, ε))`, `ε = geodesic_eps_deg`,
  and `permission` uses GMST (`geometry.greenwich_mean_sidereal_time_deg` on the CPU
  float64 `jds`). AdamW + `cosine_warmup(warmup_steps, max_steps)`, grad clip, AMP
  (bf16 CUDA / fp16 MPS). `train(cfg, num_workers, ephe_path, jpl_file, max_steps)`
  loops epochs (`set_epoch`) and checkpoints via `bundle.save_checkpoint`.
- **`bundle.py`** — `build_models(cfg)`, `save_checkpoint(...)` / `load_checkpoint(...)`
  (format `kalachakra-decoupled-v1`: both `state_dict`s + full config + optim/sched/step).
- **`inference.py`** — `DecoupledInference`: `global_texture`, `pinpoint`,
  `tension_vector`, `tension_batch` (see [07 §9](07-inference-and-analysis.md)).
- **`export.py`** — TorchScript / ONNX (see [07 §9.2](07-inference-and-analysis.md)).
- **`api.py`** — the live dashboard server (see [08 §7.4](08-api-and-serving.md)):
  `create_app(ckpt_path=None, device, bank_size=64, ephe_path, jpl_file)` with a
  demo (random‑weights) fallback; `serve(...)`.
- **`web/decoupled.html`** — the browser UI: fetches `/api/texture` per frame, decodes
  the OKLab→sRGB buffer onto a canvas, overlays coastlines + graticule, animates the
  timeline (play/pause, JD scrubber, step select, resolution, opacity), and
  click‑to‑inspect via `/api/point` (colour swatch + per‑planet attribution bars).

---

## 8. CLI (`__main__.py` + scripts)

```bash
# training (full flag surface on the script)
python scripts/train_decoupled.py --timeline-start 2024-01-01 --timeline-end 2025-01-01 \
    --steps 20000 --batch 8 --out-dir checkpoints/decoupled

# unified module CLI
python -m kalachakra.decoupled_engine train  --steps N --batch B [--timeline-start …]
python -m kalachakra.decoupled_engine eval   CKPT [--lat --lon | texture]
python -m kalachakra.decoupled_engine export CKPT --format torchscript|onnx|both
python -m kalachakra.decoupled_engine serve  [CKPT]     # omit CKPT → demo mode

# live dashboard
python scripts/serve_decoupled.py --checkpoint CKPT   # or no --checkpoint for demo
```

---

## 9. End‑to‑end shape trace

```
jd ──► celestial_features (10,5) ──► SkyEncoder ──► tension (512)
                                                       │
(lat,lon) grid (P,2) ──► latlon_to_unit_vector (P,3) ──► FourierFeatures (P,128)
                                                       │ concat tension
                                                       ▼
                                          Earth Lens MLP ──► OKLab (P,3)
                                                       ▼
                                          OKLab→sRGB8 ──► WebGL texture bytes
```

All grounded in continuous differential geometry and wave mechanics — no arbitrary
heuristics, no static spatial discretization.
