# 03 · Grid, Projection & Storage (Phase 2)

How the global state becomes a per‑location field, and how the timeline is stored
and streamed. Modules: `grid/geodesic.py`, `projection/{spatial,microgrid}.py`,
`geo/h3index.py`, `storage/{binary_store,ring_buffer,mipmap,parquet_store,duckdb_engine}.py`.

---

## 1. `grid/geodesic.py` — the observer mesh

A `Grid` is a spherical observer mesh with per‑node latitude/longitude (radians)
and unit `xyz`:

```python
class Grid:  lat: (N,)  lon: (N,)  xyz: (N,3)   @property n_nodes
```

Constructors:

- `fibonacci_sphere(n)` — a spherical Fibonacci lattice of exactly `n` near‑uniform
  points. **This is how the canonical 122,880 nodes are realized** (`default_grid()`),
  because a subdivided icosphere yields `10·4ⁿ+2` vertices, which never equals
  122,880.
- `icosphere(subdivisions)` — literal recursive icosahedral subdivision;
  `icosphere_vertex_count(s) = 10·4ˢ + 2`.
- `default_grid()` → `fibonacci_sphere(N_SPATIAL_NODES)`.
- `_latlon_from_xyz(xyz)` — inverse.

Distance on the mesh is angular separation (great‑circle), never Euclidean lat/lon.

---

## 2. `projection/spatial.py` — `G(t) → E(t, s)`

The Phase‑2 heart: closed‑form spherical trigonometry that projects the global
state onto every observer node in **one broadcast**, with no per‑node ephemeris
call. The numpy `project()` is the correctness oracle for any Metal/GPU kernel.

**Local field encoding.** For node `s` and body `i`:

```
e_i(s,t) = [ cosθ·cosh,  sinθ·cosh,  sinh,  cosΔφ,  sinΔφ ]
           θ = azimuth,  h = altitude,  Δφ = body longitude − Ascendant
```

so the field is `(N, 10, 5)` → flattened `(N, 50) = LOCAL_FIELD_WIDTH`.

**`project(global_frame, jd_ut, grid) → (N, 10, 5)`** steps:

1. Decode each body's geocentric ecliptic direction & distance from `G(t)`
   (`decode_ecliptic`).
2. Rotate to equatorial; form each body's geocentric position vector in AU.
3. Compute per‑node local sidereal time → RAMC, and the observer's
   surface offset vector `ρ = EARTH_RADIUS_AU · (cosφcosRAMC, cosφsinRAMC, sinφ)`.
4. **Topocentric subtraction:** `r_topo = r_body − ρ`, applied **only to the seven
   physical bodies** (a parallax mask; the nodes and Ayanamsha are directions at
   infinity). This resolves lunar parallax (~0.95°) so an eclipse localizes to its
   real ground track.
5. From `r_topo`: topocentric RA/Dec → hour angle `H = LST − RA` → altitude `h`,
   azimuth `θ`. Topocentric ecliptic longitude → `Δφ = lon_topo − Ascendant`.
6. Assemble the 5‑vector per (node, body). Asserts shape `(N, N_BODIES, 5)`.

**`ascendant_longitude(ramc, ε, geo_lat) → radians`** — the standard closed form for
the rising ecliptic point from RAMC, obliquity and latitude; vectorizes over both
RAMC and latitude (used heavily by the Kundali engine and the smoothness loss).

`decode_ecliptic(global_frame) → (lon, lat)` recovers ecliptic angles from the
first three columns of `G(t)`.

The field's meaning is versioned by `PROJECTION_VERSION` (currently 2 = topocentric);
`1` was geocentric‑only. Checkpoints and indexes stamp it and warn on mismatch.

---

## 3. `projection/microgrid.py` — regional LOD

On‑the‑fly regional grids for the dynamic level‑of‑detail engine (no fixed
discretization):

- `bbox_microgrid(min_lat, min_lng, max_lat, max_lng, density=64) → Grid` — a dense
  lat/lon patch as a `Grid` for a bounding box.
- `resolution_km(min_lat, min_lng, max_lat, max_lng, density)` — the ground
  resolution of such a patch, so a 5‑km viewport can be requested directly.

---

## 4. `geo/h3index.py` — Uber H3 geospatial indexing

Hexagonal spatial indexing for the persisted token datasets (optional `h3` dep;
degrades to a lat/lon‑bucket fallback):

- `cell_for(lat, lng, resolution=BASE_RESOLUTION)`, `cells_for_grid(...)` — H3 cell id(s).
- `parent(cell, resolution)`, `neighbors(cell, k=1)`, `cells_in_bbox(...)`.
- `h3_available()`, `_fallback_cell(...)`.

H3 cells become a spatial predicate in the DuckDB router (§7).

---

## 5. `storage/binary_store.py` — BF16 memmap timeline

Serializes `G(t)` to contiguous `.mmap` chunks in **BF16** with **temporal delta
encoding**.

- `float32_to_bf16(x)` / `bf16_to_float32(b)` — round‑trip via truncation (uint16
  view of the top 16 bits).
- `delta_encode(frames)` / `delta_decode(deltas)` — store frame‑to‑frame differences
  (smooth, highly compressible).
- `ChunkMeta` — sidecar metadata per chunk (start frame, count, dtype, delta flag).
- `EphemerisStore` — reader/writer:
  - `write_chunk(start_frame, frames, delta_encoded=True)` + manifest append.
  - `read_chunk(start_frame) → (N, 10, 7)` float32.
  - `chunks() → list[ChunkMeta]`.

BF16 halves the float32 footprint to ~1.9 TB for the full timeline; the delta stream
is the path to the ~300 GB target with added entropy coding.

---

## 6. `storage/ring_buffer.py` — async prefetch

`RingBuffer(store, start_frames, max_prefetch=3)` — a background worker thread that
decodes upcoming chunks ahead of the consumer and yields them in timeline order,
overlapping CPU decode with GPU compute. Context‑managed (`start()`, `close()`,
`_worker()`). This is the 20 GB budget line in the memory partition and the source
that `data.dataset.EphemerisStream` streams from.

---

## 7. Token persistence & query (Phase 4 index)

For the tokenized/served index (`build_index.py`), three storage modules:

### 7.1 `storage/mipmap.py` — three‑tier temporal mipmap

Roll the native per‑frame token stream up into coarser tiers so wide time spans
don't page in billions of rows:

- Bucket reducers: `bucket_max/mean/std`, `mode_per_bucket(tokens, bucket, n_tokens)`.
- `hourly_rollup(potential, shear, leaf, n_tokens=4096)` — tier‑2.
- `daily_rollup(potential, shear, rarity, leaf, rarity_threshold=0.9, …)` — tier‑3,
  keeping only rarity‑thresholded days (epochal).
- `select_tier(span_frames, target_rows=1000)` → `"tier1"|"tier2"|"tier3"`.

### 7.2 `storage/parquet_store.py` — partitioned Parquet

`ParquetTokenStore` writes century‑partitioned Apache Parquet for the three tiers:

- `write_frames / write_hourly / write_daily(columns)`; `tier_glob(tier)`, `has_tier(tier)`.
- `century_of(jd)` partitions by century; `read_meta()/write_meta(**fields)` and
  `projection_version()` stamp compatibility.

### 7.3 `storage/duckdb_engine.py` — viewport query router

`DuckDBEngine` runs an in‑process DuckDB over the Parquet datasets:

- `ViewportQuery(bbox, time span, limit, …)`.
- `query(q)` — routes to the optimal tier (`_tier_for`), builds an H3/lat‑lon spatial
  predicate (`_spatial_predicate`), returns bounded rows (~24 ms viewport queries).
- `token_pmf(tier)` — empirical token distribution (feeds the Rarity Index).

Together these back the kinetic‑radar serving app (`serving.app`, [08](08-api-and-serving.md)).

---

## 8. Data flow through this layer

```
ephemeris.global_state ──► G(t) (10×7)
        │                     │
        │ (store path)        │ (live path)
        ▼                     ▼
 binary_store (.mmap BF16 Δ)  projection.spatial.project ──► E(t,s) (N×50)
        │                            ▲
        ▼                            │
 ring_buffer ──► data.dataset.EphemerisStream ──────────────┘  (windows for training)
        │
        ▼  (after tokenization, build_index.py)
 mipmap ──► parquet_store (tier1/2/3) ──► duckdb_engine ──► serving
```
