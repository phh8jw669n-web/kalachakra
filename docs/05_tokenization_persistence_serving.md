# 05 — Tokenization, deep-time rarity, persistence, and serving

This is the "next phase" pipeline built on top of the latent autoencoder: turn the
continuous 64-d latent into discrete geometric tokens, quantify deep-time rarity,
persist everything for millisecond queries, and stream it to a kinetic client.

```
E(t,s) --encoder--> z(64) --RVQ--> (macro,micro) tokens + quantized z
   --> rarity index (deep-time NLL)
   --> Parquet (century-partitioned, tier1/2/3) indexed by H3 + time
   --> DuckDB router (tier + partition prune + H3 + rarity)
   --> FastAPI /inspect (JSON) + /stream (binary) --> WebGL kinetic radar
```

## Unsupervised discretization (§2) — `kalachakra.models.rvq`

`HierarchicalResidualVQ` snaps each latent to a **macro** code (64 broad
archetypes), then snaps the macro residual within a **micro** codebook
*conditioned on the macro index* (64 per macro → **4096 leaf tokens**). Standard
VQ-VAE machinery: straight-through gradients, commitment loss (β = 0.25), EMA
codebook updates, and dead-code replacement against collapse. `leaf = macro*64 +
micro`. `QuantizedSphericalAutoencoder` wires encoder → RVQ → decoder.

## Deep-time rarity (§2) — `kalachakra.analysis.rarity`

`RarityModel` keeps the empirical PMF over the 4096 tokens across the corpus. The
**Rarity Index** is the normalized negative log-likelihood of a token: ~0 for
configurations common across ten millennia, ~1 for those seen a handful of times.
`significance_threshold(q)` gives the count-weighted rarity cutoff the radar
tightens toward 99.99 % at deep-time zoom.

## Token serialization (§2) — `kalachakra.analysis.tokens`

Compact 4-byte `(macro, micro)` uint16 tokens; full per-node descriptors add the
rarity and the 64-d latent; `to_columns()` produces Parquet-ready columnar arrays.

## Persistence (§3) — `kalachakra.storage`, `kalachakra.geo`

- **`geo.h3index`** — Uber H3 (v4) maps nodes to int64 cell ids at resolution 4
  (localized res 7); bbox polygon fill + neighbors for constant-time spatial sets.
- **`storage.mipmap`** — three tiers: native 24 s frames, hourly (150-frame)
  rollups (max potential / peak shear / modal archetype), daily rollups
  (stats + anomaly counts). `select_tier` bounds any scan to ~1000 rows.
- **`storage.parquet_store`** — century-partitioned Parquet, dictionary-encoded
  tokens, Snappy floats; tier1/2/3 datasets.
- **`storage.duckdb_engine`** — in-process DuckDB router: tier selection from
  span/velocity, century partition pruning, H3 set filtering for regional boxes
  (lat/lng range fallback for large/global spans), rarity thresholding. Verified
  ~24 ms viewport queries over a real index.

## Dynamic radar (§6) — `kalachakra.analysis.radar`

`temporal_stride` decimates to a bounded point count; `band_gains` is the
four-band spectral mixer (micro/fast/cyclic/macro); `significance_percentile`
scales the alert threshold with span; `build_news_card` emits the textless,
pure-geometry event payload.

## Serving (§7) + client (§5) — `kalachakra.serving`, `web/radar.html`

- **`serving.binary`** — little-endian `KCHR` field frames for zero-copy WebGL.
- **`serving.app`** — FastAPI `/health`, `/inspect` (JSON control plane via the
  DuckDB engine), and `/stream` (binary WebSocket). Run with
  `scripts/serve_radar.py`.
- **`web/radar.html`** — Three.js multi-channel client: potential → luminance,
  VQ archetype → hue LUT (tension warm / harmony cool / singularity bright),
  particle advection from potential-gradient vorticity scaled by shear, and the
  diurnal terminator sweep. Parses the binary stream; falls back to a synthetic
  field so it renders without a server.

## End-to-end

```bash
pip install -e ".[train,index,serve]"          # torch + pyarrow/duckdb/h3 + fastapi
python scripts/build_index.py --out data/index --nodes 256 --frames 3000
python scripts/serve_radar.py --index data/index --port 8000
# open web/radar.html, point its WebSocket box at ws://127.0.0.1:8000/stream
```

`build_index.py` streams real ephemeris → projects → quantizes → computes rarity
→ writes tier-1 Parquet + hourly rollups. With a trained checkpoint the tokens
are meaningful; without one the full data path still runs (untrained tokens).

## Status

Every module here is implemented and unit-tested (130 tests total). The RVQ,
rarity, tokens, mipmap, H3, Parquet/DuckDB, binary framing, and FastAPI
control-plane/WebSocket paths are exercised end to end in the sandbox
(300 real frames × 64 nodes → queryable index; ~24 ms queries). The WebGL client
is standard Three.js (not unit-testable here) and renders from the binary stream
or a synthetic fallback. Training the codebook to convergence and running the
full 13.4-billion-frame offline inference are the compute-bound steps that run on
the target hardware.
