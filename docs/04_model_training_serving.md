# 04 — Neural core, training, analysis, and serving

## Model (`kalachakra.models`)

`SphericalAutoencoder` composes `n_blocks` spatio-temporal blocks. Each `STBlock`:

1. **Spatial** — `GeodesicConv`: message passing over precomputed geodesic
   `k`-NN neighborhoods (`build_knn`), preserving great-circle relationships with
   no map projection.
2. **Temporal** — `FourierBlock1d`: a spectral `SpectralConv1d` (learned complex
   modes, stored as a *real* `(…, 2)` parameter so every optimizer works) plus a
   pointwise residual.

Tensor flow: `E (B,T,N,50) → lift → blocks → 64-d latent z(t,s) → blocks →
recon (B,T,N,50)`. Each latent code summarizes a temporal window and spatial
neighborhood, so 64 dims is a genuine bottleneck on the receptive volume.

> **Mixed precision + FFT.** The spectral path forces float32 with autocast
> disabled (bf16/fp16 FFTs are unsupported), then casts back — so bf16 training
> and spectral learning coexist. This is exercised by `tests/test_trainer.py`.

## Loss (`kalachakra.losses`)

`CompositeGeodesicLoss` = weighted sum of:

- **geodesic reconstruction** — clamped-arccos great-circle error;
- **spectral harmonic** — amplitude + phase divergence of the temporal spectrum;
- **aspect relational invariance** — divergence of pairwise angular-separation
  matrices (rotation-invariant).

`losses.reference` is the numpy oracle (tested, incl. rotation invariance);
`losses.geometric` is the differentiable torch version.

## Training (`kalachakra.training`)

- **`Lion`** optimizer (single momentum buffer; AdamW fallback) with cosine
  annealing + warm restarts (`build_scheduler`).
- **`Trainer`** — device auto-select (MPS → CUDA → CPU), bf16 autocast, grad
  clipping, and two checkpoint tiers: `micro_*` every 12 h, `era_*` every 500
  simulated years.
- **`TestingDaemon`** — a separate process/loop watching for `era_*` snapshots,
  scoring each against benchmark events (eclipses, great conjunctions, synthetic
  resonance) and logging the **Resonance Divergence Metric** — never pausing the
  GPU loop.

## Analysis (`kalachakra.analysis`)

With the decoder severed, work on `z(t, s)`:

- `signatures.geometric_potential_field` = `‖z‖`;
- `signatures.temporal_shear_gradient` = `‖dz/dt‖`;
- `clustering.cluster_latents` — HDBSCAN (with a labeled dependency-free
  fallback);
- `anomaly.detect_singularities` — robust median+MAD thresholding where both
  fields spike simultaneously.

## Serving (`kalachakra.serving` + `web/`)

`BroadcastEngine` answers point queries (nearest geodesic node → potential,
shear, cluster) and emits a full-mesh heatmap. `serving.api.create_app` exposes
`/potential` and `/heatmap` over FastAPI; `web/index.html` renders the mesh as a
Three.js globe with potential-driven vertex displacement and heat coloring,
falling back to a synthetic field when the API is offline.

```bash
python scripts/serve.py --demo --nodes 4096 --port 8000
# open web/index.html and point its API box at http://127.0.0.1:8000
```
