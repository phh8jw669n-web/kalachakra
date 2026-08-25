# 01 · Architecture Overview

This document maps the whole system: the layered module structure, the two neural
pipelines, and the invariants every layer obeys. Deep detail lives in the
per‑topic files; this is the map that ties them together.

---

## 1. Layered structure

The codebase is layered so each tier depends only on the ones below it. The bottom
three layers are pure numpy (+ `pyswisseph`) and require no torch; everything ML
sits above.

```
  ┌──────────────────────────────────────────────────────────────────────┐
  │  SERVING / UI      serving/ (REST, gRPC, WS)  ·  web/*.html  ·  cli.py │
  ├──────────────────────────────────────────────────────────────────────┤
  │  APPLICATIONS      indexer/ (Great Indexer)  ·  kundali/  ·            │
  │                    decoupled_engine/  ·  analysis/  ·  transducer/     │
  ├──────────────────────────────────────────────────────────────────────┤
  │  ML CORE           models/  ·  losses/  ·  training/  ·  data/         │
  ├──────────────────────────────────────────────────────────────────────┤
  │  SPATIAL / STORAGE grid/  ·  projection/  ·  geo/  ·  storage/         │
  ├──────────────────────────────────────────────────────────────────────┤
  │  FOUNDATIONS       constants.py  ·  geometry.py  ·  ephemeris/         │
  └──────────────────────────────────────────────────────────────────────┘
       numpy + pyswisseph only  │  torch above the ML CORE line
```

| Layer | Modules | Requires | Doc |
|---|---|---|---|
| Foundations | `constants`, `geometry`, `ephemeris/*` | numpy, pyswisseph | [02](02-ephemeris-and-geometry.md) |
| Spatial / storage | `grid`, `projection`, `geo`, `storage` | numpy (+ pyarrow/duckdb/h3 optional) | [03](03-projection-grid-storage.md) |
| ML core | `models`, `losses`, `training`, `data` | torch | [05](05-models.md), [06](06-training.md), [04](04-data-pipelines.md) |
| Applications | `indexer`, `kundali`, `decoupled_engine`, `analysis`, `transducer` | mixed | [07](07-inference-and-analysis.md), [09](09-decoupled-engine.md), [10](10-kundali-engine.md) |
| Serving / UI | `serving`, `web`, `cli` | fastapi/grpc/uvicorn optional | [08](08-api-and-serving.md) |

---

## 2. The two neural architectures

The project contains **two distinct ways** to turn `G(t)` into an energy field. They
share the foundations but diverge completely above them.

### 2.1 Discrete VQ‑mesh pipeline (original)

The blueprint architecture. Four phases:

| Phase | What happens | Modules |
|---|---|---|
| **1** | Compute `G(t)` (10×7) from the ephemeris; optionally serialize to a BF16 `.mmap` store | `ephemeris.global_state`, `storage.binary_store` |
| **2** | Analytically project `G(t)` onto 122,880 mesh nodes → `E(t,s)` (10×5 per node); stream via ring buffer | `projection.spatial`, `grid.geodesic`, `storage.ring_buffer`, `data.dataset` |
| **3** | Compress `E(t,s)` with a Spherical Autoencoder + Spatio‑Temporal FNO → 64‑D latent; optionally quantize to 4096 archetypes | `models.autoencoder*`, `models.fno`, `models.spherical_conv`, `models.rvq`, `training/*`, `losses/*` |
| **4** | Derive potential/shear signatures, cluster, detect singularities, persist to Parquet/DuckDB, serve to WebGL | `analysis/*`, `storage.parquet_store`, `storage.duckdb_engine`, `serving/*` |

On top of Phase 3–4 sit two large subsystems:

- **The Great Indexer** (`indexer/`) — profiles all 4096 archetypes into 18
  mathematical metrics across 5 domains and compiles a `dossiers.sqlite`. See
  [07 §3](07-inference-and-analysis.md).
- **The Isomorphic Transducer** (`transducer/`) — a losslessly invertible,
  physics‑based renderer of the latent field. See [07 §2](07-inference-and-analysis.md).

### 2.2 Continuous decoupled engine (newer)

`decoupled_engine/` replaces the mesh **and** the codebook with a continuous field:

```
  celestial tensor (B,10,5)  ──►  Sky Encoder (transformer)  ──►  tension vector (B,512)
                                                                      │
   (lat,lon) any resolution  ──►  Earth Lens (implicit MLP)  ◄────────┘   ──►  OKLab (…,3)
```

No 122,880 nodes, no 4096 tokens — the field is evaluated on demand at any
`(lat, lon)`. Trained self‑supervised from wave mechanics and spherical geometry.
Full spec in [09](09-decoupled-engine.md).

### 2.3 How they relate

| | Discrete VQ‑mesh | Decoupled engine |
|---|---|---|
| Spatial representation | fixed 122,880‑node mesh | continuous, query any point |
| Global summary | 64‑D latent per node (+ 4096 VQ tokens) | one 512‑D tension vector |
| Bodies modeled | Sun…Saturn + Rahu/Ketu/Ayanamsha | Sun…Pluto (adds outer planets) |
| Output | potential/shear scalars, tokens | OKLab perceptual color field |
| Training signal | reconstruction (geodesic+spectral+aspect) | contrastive + smoothness + temporal |
| Best for | archetype mining, rarity, tokenized index | live infinite‑resolution visualization |

Both are first‑class; the decoupled engine is the most recent and the basis of the
live global dashboard.

---

## 3. The data objects (canonical shapes)

These shapes recur throughout the code. Widths come from `constants.py`.

| Object | Shape | Meaning |
|---|---|---|
| `G(t)` global frame | `(10, 7)` | per body: `[cosλcosβ, sinλcosβ, sinβ, λ̇, β̇, r, ṙ]` |
| `G(t)` batch | `(T, 10, 7)` | `T` frames |
| Local field `E(t,s)` | `(N, 10, 5)` → flattened `(N, 50)` | per node: `[cosθcosh, sinθcosh, sinh, cosΔφ, sinΔφ]` |
| Training window (mesh) | `(T, N, 50)` | `T`‑frame temporal window over `N` nodes |
| Latent `z(t,s)` | `(…, 64)` | continuous bottleneck (`LATENT_DIM`) |
| VQ tokens | macro `(…,)`, micro `(…,)`, leaf `= macro*64 + micro` | 64×64 = 4096 archetypes |
| Celestial tensor (decoupled) | `(B, 10, 5)` | per body: `[sinλ, cosλ, sinβ, cosβ, λ̇]` |
| Tension vector | `(B, 512)` | Sky Encoder output (L2‑normalized) |
| Color field | `(M, P, 3)` | Earth Lens OKLab per point |

Key constants (`kalachakra.constants`):

```
KALI_YUGA_EPOCH_JD = 588465.5      TIMELINE_YEARS = 10256   (end 7154 CE)
VIGHATIKA_SECONDS  = 24            HORIZON_ADVANCE_DEG_PER_FRAME ≈ 0.10027
N_SPATIAL_NODES    = 122880        LATENT_DIM = 64          N_BODIES = 10
GLOBAL_BODY_FEATURES = 7  →  GLOBAL_STATE_WIDTH = 70
LOCAL_BODY_FEATURES  = 5  →  LOCAL_FIELD_WIDTH  = 50
PROJECTION_VERSION = 2   (topocentric, parallax on physical bodies only)
total_temporal_frames() ≈ 1.349e10 ;  raw_uncompressed_bytes() ≈ 1.9 TB (BF16)
```

---

## 4. Design invariants (enforced across layers)

1. **Native units.** Time in Vighatikas (24 s); space in angular separation on a
   geodesic mesh. No 60‑minute hour, no 360° grid. (`constants`, `grid.geodesic`.)
2. **Boundary‑free encodings.** Every cyclic angle enters as `(cos, sin)` or a unit
   vector — no 0/360° discontinuity. (`ephemeris.encode_body`, `features.encode_celestial`.)
3. **Global/local decoupling.** `G(t)` computed once; the local field is a pure
   broadcast/query with no per‑observer ephemeris call. (`projection.spatial`,
   `decoupled_engine`.)
4. **Objective topology.** Outputs are geometric quantities (separations, resonance,
   tension, latent norms, colors) — never event predictions or interpreted text.
5. **Versioned projection semantics.** `PROJECTION_VERSION` is stamped into
   checkpoints/indexes so incompatible artifacts refuse to mix. (`training.checkpoint`,
   `storage.parquet_store`.)
6. **Reference‑oracle testing.** Numpy reference implementations
   (`losses.reference`, `projection.spatial`) are the correctness oracles for the
   torch/GPU paths, checked bit‑for‑bit within BF16 tolerance.

---

## 5. Backends and the timeline

`G(t)` is produced by the Swiss Ephemeris (`pyswisseph`) with a selectable backend:

| Backend | Coverage | Data files | Selected by |
|---|---|---|---|
| **Moshier** (default) | ~3000 BCE – 3000 CE | none | default |
| **Swiss** (`.se1`) | full 10,256‑yr span | 36 `.se1` files (~40 MB, DE431) | `configure(mode="swiss", ephe_path=…)` |
| **JPL** (DE441) | full span | DE441 `.bsp` kernels | `configure(mode="jpl", jpl_file=…)` |

`scripts/setup_full_span.py` downloads/verifies the `.se1` files and writes a config
so every command uses the full span automatically. Everything demoable in this repo
runs on the file‑free Moshier backend; the full 3102 BCE→7154 CE span needs Swiss or
JPL. See [11 §5](11-cli-and-configuration.md).

---

## 6. Test topology

The suite (`tests/`, ~223 tests) mirrors the layers: `test_constants`,
`test_geometry`, `test_calendar`, `test_ephemeris_*`, `test_grid`, `test_projection`,
`test_microgrid`, `test_storage`, `test_persistence`, `test_torch_model`,
`test_autoencoder_v2/v3`, `test_quantized_ae`, `test_rvq`, `test_losses`,
`test_trainer`, `test_checkpoint`, `test_analysis`, `test_weather`, `test_rarity`,
`test_tokens`, `test_radar`, `test_resonance`, `test_transducer`, `test_serving_app`,
`test_grpc_server`, `test_great_indexer`, `test_curriculum`, `test_kundali`,
`test_decoupled_engine`, `test_cli`, `test_se1_files`, `test_setup_script`,
`test_profile_archetypes`. Ephemeris/torch/fastapi‑dependent tests skip gracefully
when the optional dependency is absent.
