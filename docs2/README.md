# Project Kalachakra — Documentation (docs2)

> An autonomous, unsupervised ML system that maps the continuous spatio‑temporal
> geometric waves of the solar system onto a model of Earth, and broadcasts
> objective mathematical **energy signatures** — structural tension, harmonic
> resonance, geometric singularities — rather than text.

This `docs2/` set is a complete, self‑contained reference for the **entire**
project: both neural architectures (the discrete VQ‑mesh pipeline and the newer
continuous *decoupled* engine), the ephemeris/geometry foundations, the data
pipelines, training mechanics, inference/analysis engines, the Great Indexer, the
Kundali Twin search engine, and every serving/API surface and dashboard.

It is written for two audiences at once:

1. **Complete technical reference** — exhaustive module/tensor/loss/hyperparameter
   detail, precise enough to rebuild or extend the project from scratch.
2. **Intuitive conceptual guide** — the philosophy, the celestial↔terrestrial
   decoupling, and what a user experiences across macro and micro timelines.

---

## How to read this

| If you want… | Start here |
|---|---|
| The big idea & philosophy | [`00-conceptual-guide.md`](00-conceptual-guide.md) |
| A map of all modules and both pipelines | [`01-architecture-overview.md`](01-architecture-overview.md) |
| Ephemeris, `G(t)`, constants, calendar, geometry | [`02-ephemeris-and-geometry.md`](02-ephemeris-and-geometry.md) |
| The mesh, `G(t)→E(t,s)` projection, storage tiers | [`03-projection-grid-storage.md`](03-projection-grid-storage.md) |
| Datasets, streaming, curriculum learning | [`04-data-pipelines.md`](04-data-pipelines.md) |
| Every model (autoencoders, FNO, VQ/RVQ, Sky/Earth) | [`05-models.md`](05-models.md) |
| Losses, optimizers, trainer, checkpoints, daemon | [`06-training.md`](06-training.md) |
| Weather engine, transducer, Great Indexer, inference | [`07-inference-and-analysis.md`](07-inference-and-analysis.md) |
| REST / gRPC / WebSocket / dashboard APIs | [`08-api-and-serving.md`](08-api-and-serving.md) |
| The continuous decoupled engine (deep dive) | [`09-decoupled-engine.md`](09-decoupled-engine.md) |
| The Kundali Twin sidereal search engine | [`10-kundali-engine.md`](10-kundali-engine.md) |
| CLIs, scripts, config dataclasses, extras | [`11-cli-and-configuration.md`](11-cli-and-configuration.md) |
| Terminology | [`12-glossary.md`](12-glossary.md) |

---

## The project in one page

Kalachakra treats planetary orbits as **continuous wave generators**. It computes
the global ephemeris state `G(t)` (ten bodies × seven features) at any instant
over an exact **10,256‑year** timeline, and then follows one of two paths:

- **Discrete VQ‑mesh pipeline** (the original architecture): analytically project
  `G(t)` onto **122,880** observer nodes to form the local field `E(t,s)`, compress
  it through a **Spherical Autoencoder + Spatio‑Temporal Fourier Neural Operator**
  into a 64‑D latent, quantize into **4096** discrete geometric archetypes, and
  serve per‑coordinate potential/shear metrics to a WebGL globe. On top sit the
  **Great Indexer** (18 mathematical profiles per archetype across 5 domains), a
  deep‑time **Rarity Index**, Parquet/DuckDB persistence, and the **isomorphic
  transducer** (a losslessly invertible physics‑based renderer).

- **Continuous decoupled engine** (`decoupled_engine/`, the newer architecture):
  drop the mesh and the codebook entirely. A transformer **Sky Encoder** compresses
  the ten‑body state into one 512‑D **global tension vector**; an implicit
  **Earth Lens Decoder** maps that vector plus any continuous `(lat, lon)` to a
  perceptual **OKLab** color — an energy field queryable at infinite resolution and
  animated live across the globe.

Both are grounded in the same invariants: native time in **Vighatikas** (24 s),
boundary‑free `(cos, sin)` angle encodings, global/local decoupling, and objective
topology over narrative.

---

## Repository quick map

```
src/kalachakra/
  constants.py           canonical figures (timeline, units, memory budget)
  geometry.py            numpy geodesic/astronomy primitives
  cli.py                 the `kalachakra` command (reading / map / scan)
  ephemeris/             G(t): bodies, timeline, calendar, se1_files, backends
  grid/                  geodesic Earth mesh (fibonacci + icosphere)
  projection/            analytical G(t) -> E(t,s); regional micro-grids
  geo/                   Uber H3 hexagonal geospatial indexing
  storage/               BF16 mmap + ring buffer; temporal mipmap; Parquet; DuckDB
  data/                  streaming datasets (EphemerisStream, curriculum)
  models/                spherical conv, 1-D FNO, autoencoders v1/v2/v3, RVQ
  losses/                composite geodesic loss (numpy reference + torch)
  training/              Lion optimizer, trainer/checkpoints, testing daemon
  analysis/              weather, signatures, clustering, anomaly, rarity, radar
  serving/               broadcast engine, binary framing, FastAPI + WS, gRPC
  transducer/            isomorphic transducer (SH topography, optics, LIC)
  indexer/               the Great Indexer (18 profiles / 5 domains -> SQLite)
  kundali/               Kundali Twin sidereal search engine
  decoupled_engine/      continuous Sky Encoder + Earth Lens projection engine
scripts/                 CLIs: train*, serve*, build_index, great_indexer, setup...
web/                     WebGL/canvas dashboards (index, radar, resonance, kundali,
                         decoupled)
configs/default.yaml     operator knob board
proto/kalachakra.proto   gRPC CosmicWeather contract
```

See the older per‑phase notes in [`../docs/`](../docs/) and the blueprint framing
in [`../ARCHITECTURE.md`](../ARCHITECTURE.md); this set supersedes and extends them
with the modules added since (the decoupled engine, curriculum learning, the Great
Indexer, and the Kundali engine).
