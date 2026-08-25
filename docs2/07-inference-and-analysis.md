# 07 · Inference & Analysis

Everything that turns geometry or a trained latent into interpretable output: the
objective weather engine, latent‑manifold analysis, tokenization/rarity, the
isomorphic transducer, the Great Indexer, and decoupled inference/export.

Modules: `analysis/*`, `transducer/*`, `indexer/*`, `decoupled_engine/{inference,export}.py`.

---

## 1. `analysis/weather.py` — objective signatures (no model)

The cosmic‑weather engine derives geometric energy signatures directly from `G(t)`
— **no trained model required**. This is what `kalachakra reading`/`scan` and the
serving app expose.

| Function | Output |
|---|---|
| `separations_deg(lons_rad)` | pairwise angular separations (deg) |
| `aspect_field(lons_rad, weights, orb)` | **harmonic resonance** — weighted aspect kernel over conjunction/opposition/trine/square/sextile |
| `stellium_concentration(lons_rad, weights)` | **geometric potential** — circular concentration (mass convergence) |
| `eclipse_state(lons_rad, orb_deg=12)` | Sun–Moon + Moon–node proximity → eclipse flags |
| `dominant_aspects(lons_rad, …, top=5)` | strongest exact aspects with strengths |
| `stations(global_frame, threshold_deg_per_day=0.05)` | retrograde stations (speed sign change) |
| `frame_signature(jd_ut, orb)` → `FrameSignature` | the full objective weather for one instant |
| `temporal_shear(jd_ut, dt_days=1/24, orb)` | frame‑to‑frame change (**structural tension**) |
| `local_intensity(field, activation, weights)` | per‑node intensity from the projected field |
| `weather_map(jd_ut, grid, dt_days, orb)` | per‑node weather over a mesh |

`FrameSignature` bundles harmonic resonance, structural tension, geometric potential,
eclipse proximity, dominant aspects, and stations. `BODY_WEIGHTS` / `DEFAULT_ORB_DEG`
are module constants.

Validated against real events (2024‑04‑08 eclipse: Sun–Moon 0.04°).

---

## 2. `analysis/signatures.py` — latent energy signatures

With a trained model's decoder severed, analysis runs on the latent `z(t,s)`:

- `geometric_potential_field(z, axis=-1)` = `‖z‖` — mass‑convergence potential.
- `temporal_shear_gradient(z, time_axis=0, dt=1, latent_axis=-1)` = `‖dz/dt‖` — phase
  transitions.
- `energy_signature(z, …)` — both together (the served scalar pair).

## 3. `analysis/clustering.py` — manifold clustering

`cluster_latents(z, min_cluster_size=50, min_samples=None) → ClusterResult` — HDBSCAN
over the latent manifold groups recurring interference patterns into objective
clusters. `hdbscan_available()` guards the optional dep; `_fallback_cluster` provides
a deterministic fallback. (`configs/default.yaml: hdbscan_min_cluster_size = 50`.)

## 4. `analysis/anomaly.py` — singularity detection

- `robust_threshold(field, sigma=4.0)` — median + `sigma·MAD` robust cutoff.
- `detect_singularities(potential, shear, sigma=4.0, max_events=None) → [Singularity]`
  — coordinates of maximum structural tension. (`singularity_sigma = 4.0`.)

---

## 5. Tokenization & rarity

### 5.1 `analysis/tokens.py`

Composite token descriptors + serialization for the persisted index:

- `leaf_id(macro, micro, n_micro=64) = macro·64 + micro`; `split_leaf(leaf)` inverse.
- `pack_tokens(macro, micro)`, `descriptor_dtype(latent_dim=64)`,
  `build_descriptors(macro, micro, rarity, latent)`, `to_columns(descriptors)` — the
  columnar layout written to Parquet.

### 5.2 `analysis/rarity.py` — deep‑time rarity

`RarityModel` — an empirical token PMF + normalized‑NLL rarity over the 4096‑leaf
alphabet:

```
fit(token_indices) / update(...)          → build/extend the PMF
rarity(token_indices) = normalized NLL    → 0 (common) … 1 (unique)
significance_threshold(q) / percentile_threshold(rarities, q)
```

A token stream from deep history calibrates "how surprising" a configuration is —
the score that gates tier‑3 daily rollups and orders the news radar.

---

## 6. `analysis/radar.py` — the textless news radar

Turns the token/rarity stream into geometry‑only "news":

- `band_energies(activation, diurnal=0)` — split activation into temporal bands.
- `temporal_stride(span_frames, target_points=1000)` — velocity scaling for a viewport.
- `band_gains(stride, sharpness, offset)` — per‑band emphasis by zoom.
- `significance_percentile(span_years, …)` — significance threshold scaled by span
  (1 yr → 95th pct … 10,256 yr → 99.99th).
- `is_applying(lons, speeds, a, b, orb)` — whether an aspect is applying vs separating.
- `NewsCard` / `build_news_card(jd, lat, lng, global_frame, macro_id, micro_id,
  rarity, …)` — a pure‑geometry event payload anchored to a coordinate
  (`to_dict()`), rarest‑first.

---

## 7. `transducer/` — the isomorphic transducer (lossless)

A physics‑based **sensory converter** (not a visualization) that maps a localized
field to a complete optical state and — uniquely — inverts it **losslessly**
(`IsomorphicTransducer.invert(transduce(...))` recovers the exact 64‑D latent to
machine precision; verified in `tests/test_transducer.py`).

| Sub‑module | Channel | Key API |
|---|---|---|
| `photometric.py` | scalar → radiant flux; rarity → Planckian colour temperature | `naka_rushton(x,k,n)` (+ inverse), `rarity_to_temperature`, `planckian_xy`, `cct_from_xy`, `blackbody_radiance` |
| `spectral.py` | four temporal bands → an orthonormal visible spectrum | `SpectralTransducer.emit(bands)` / `recover(spectrum)` over an `_orthonormal_basis` |
| `kinematics.py` | vector field → Helmholtz‑Hodge fluid + Line Integral Convolution | `helmholtz_hodge(u,v)`, `divergence`, `curl`, `line_integral_convolution(...)` |
| `topography.py` | 64‑D latent → spherical‑harmonic Earth height field | `sh_modes`, `real_sph_harm`, `make_quadrature`, `synthesize`/`analyze` (exact Gauss‑Legendre SH) |
| `state.py` | the whole optical state, forward + inverse | `OpticalState`, `IsomorphicTransducer.transduce(...)` / `.invert(state)` |

- `naka_rushton` gives **boundless** magnitude → flux (no clipping).
- Rarity is encoded as a **colour temperature** (rare = hotter/bluer).
- The four temporal bands are emitted on an **orthonormal** spectral basis, so they
  can be recovered exactly (`gram()` ≈ I).
- The latent becomes **topography** via exact spherical‑harmonic synthesis; analysis
  recovers the coefficients.

Rendered by `web/radar.html` (dual viewport: global sphere + regional micro‑canvas
+ Sidebar Inspector). Because each channel is invertible, the picture *is* the data.

---

## 8. `indexer/` — the Great Indexer

An out‑of‑core pipeline that profiles all **4096** VQ archetypes into **18
mathematical profiles across 5 domains**, compiling a single queryable
`dossiers.sqlite`. Entry: `scripts/great_indexer.py`. Resumable & crash‑safe.

### 8.1 Adaptive clock — `indexer/adaptive.py`

`AdaptiveClock(start_jd, end_jd, coarse_s, fine_s, threshold, max_fine_run, on_down,
on_up)` yields `Tick(jd, resolution_s, fine)`s that **cruise at 1‑hour steps and
downshift to 24‑second micro‑frames** when the stacked body‑direction tensor's
velocity exceeds `threshold` (`spatial_tensor(jd)`, `tensor_velocity(t0,t1,dt)`),
then stabilize back. `.stats` tracks downshift events / fine ticks.

### 8.2 The five domains / 18 profiles

| Domain | Phase | Profiles |
|---|---|---|
| **1 — tensor physics** | `phase1_physics.py` | True Magnitude (pre‑quant `‖z‖`), Dimensional Variance, Anomaly Rank (cosine isolation), **PCA Dominance** (SVD) |
| **2 — orbital** | `phase2_sweep.py` | Multivariable Attribution (ridge normal eqns), Angular Phase Harmonic, Orbital Velocity Index, Solar Alignment |
| **3 — spatial** | `phase2_sweep.py` | Latitudinal/Polar Affinity, Spatial Coherence & Dispersion (connected components), Geographic Drift Velocity |
| **4 — temporal** | `phase3_temporal.py` (DuckDB) | Persistence RLE baseline, Harmonic Periodicity (FFT), Epoch Clustering (Fano factor) |
| **5 — ecosystem** | `phase4_ecosystem.py` | Markov Transition Lineage, Global Exclusion, Adjacency‑Halo Symbiosis, Antipodal Resonance |

### 8.3 Pipeline (`indexer/pipeline.py`)

```
Phase 1  Domain-1 physics from the codebook (+ magnitude calibration sweep)
Phase 2  adaptive temporal sweep: Domain-2/3 per-token accumulators + K×K
         co-occurrence matrices; raw activations flushed to Parquet (atomic per chunk)
Phase 3  Domain-4 temporal waveforms via DuckDB over the activation Parquet
Phase 4  Domain-5 ecosystem: Markov/exclusion (DuckDB) + halo/antipode (from Phase-2 cooc)
Master   merge all domains → one wide `tokens` row per archetype + side tables
```

- **Streaming accumulators** (`Accum`) are checkpointed as one atomic `.npz` per
  chunk; a rerun resumes at the exact frame (`state.py` `StateLock`: phases done,
  chunks done, resume point).
- **`model_io.py`** loads the v3 checkpoint + grid, projects fields, tokenizes
  batches, auto‑sizes the node batch.
- **`sweep_math.py`** — per‑frame helpers: `connected_components(tokens, neighbors)`,
  `spherical_centroid`, `subsolar_point(jd)`, `great_circle_deg`.
- **`telemetry.py`** — structured logging + a live inner‑loop **heartbeat**
  (`[P2][HB]` lines: frame rate, downshifts, hardware snapshot).
- **`master_db.py`** — `write_master(...)` builds `dossiers.sqlite` atomically: a wide
  `tokens` table (dynamic columns = union of all profile keys), an `attribution` side
  table, and four ecosystem relation tables (`transitions`, `exclusion`, `symbiosis`,
  `antipode`), all indexed. `run_meta` records span/domains/profiles.

### 8.4 `--lite` mode

For rapid UI prototyping: `great_indexer.py --lite` skips **Domain 5** (no antipode
map, no adjacency‑halo/antipodal co‑occurrence, no Markov/exclusion — the sweep's
bottleneck) and **Domain‑1 PCA/SVD**; keeps Domains 1 (magnitude/variance/anomaly),
2, 3 and 4. The skipped `tokens` columns are written as **NULL** and the relation
tables created **empty**, so the SQLite schema is unchanged for the frontend.
`run_meta` records `lite=true, domains=4, profiles=13`. The Phase‑2 heartbeat reports
`LITE graph:off`.

`IndexerConfig` (`indexer/config.py`) carries every knob (`coarse_step_seconds`,
`fine_step_seconds`, `velocity_threshold`, `chunk_frames`, `heartbeat_*`, `lite`, …)
and the output paths (`parquet_dir`, `state_path`, `master_db_path`).

---

## 9. Decoupled inference & export

### 9.1 `decoupled_engine/inference.py`

`DecoupledInference(sky, earth, cfg, device)` — one Sky‑Encoder pass per timestamp;
the tension vector is shared by every terrestrial query:

- `from_checkpoint(path, device, ephe_path, jpl_file)` — load + configure ephemeris.
- `tension_vector(jd) → (1,512)`.
- **`global_texture(timestamp, width=512, height=256, chunk=65536)`** — the dense
  energy layer: evaluate the Earth Lens over an equirectangular grid; returns
  `{oklab (H,W,3), rgb8 (H,W,3) uint8, bytes, jd}` (byte buffer for WebGL).
- **`pinpoint(timestamp, lat, lon)`** — one point: `{oklab, rgb8, attribution}` where
  `attribution` = per‑planet attention weights (sum 1).
- `tension_batch(jds)` — for latent similarity search.
- `jd_from_timestamp(ts)` accepts a JD number/numeric string or ISO/"now".

### 9.2 `decoupled_engine/export.py`

- `export_torchscript(sky, earth, cfg, out_dir)` — trace both to TorchScript
  (`sky_encoder.ts.pt`, `earth_lens.ts.pt`), verified equal to eager.
- `export_onnx(...)` — ONNX with dynamic axes (needs `onnx`); Earth Lens keeps
  dynamic `points`.
- `export_from_checkpoint(ckpt, out_dir, fmt="torchscript"|"onnx"|"both")`.

Full decoupled deep‑dive: [09](09-decoupled-engine.md). Serving surfaces: [08](08-api-and-serving.md).
