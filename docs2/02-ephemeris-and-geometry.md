# 02 · Ephemeris & Geometry (Foundations)

The bottom layer. Pure numpy + `pyswisseph`, no torch. This is where real
planetary data enters the system and where the shared math primitives live.

Modules: `constants.py`, `geometry.py`, `ephemeris/{bodies,calendar,timeline,global_state,se1_files}.py`.

---

## 1. `constants.py` — the single source of truth

Every fixed figure is defined once, annotated with provenance, and (for derived
figures) recomputed so it can be audited. Imports only the standard library, so it
is safe to import anywhere. `_validate()` runs at import time and asserts the memory
partition sums to 128 GB, the widths are 70/50, and the horizon advance ≈0.1°.

Canonical values:

```python
KALI_YUGA_EPOCH_JD          = 588465.5     # 3102-02-18 BCE 00:00 UTC
TIMELINE_YEARS              = 10_256       # → TIMELINE_END_YEAR_CE = 7154
EPHEMERIS_DATASET           = "DE441"
VIGHATIKA_SECONDS           = 24           # native temporal step
SIDEREAL_DAY_SECONDS        = 86_164.0905
HORIZON_ADVANCE_DEG_PER_FRAME = 360/SIDEREAL_DAY_SECONDS*VIGHATIKA_SECONDS ≈ 0.10027
N_SPATIAL_NODES             = 122_880
EARTH_RADIUS_AU             = 6371.0088/149597870.7   # topocentric offset
PROJECTION_VERSION          = 2            # topocentric, parallax on physical bodies
N_BODIES                    = 10
GLOBAL_BODY_FEATURES        = 7  → GLOBAL_STATE_WIDTH = 70
LOCAL_BODY_FEATURES         = 5  → LOCAL_FIELD_WIDTH  = 50
LATENT_DIM                  = 64
DAYS_PER_YEAR = 365.25   SECONDS_PER_DAY = 86_400   BF16_BYTES = 2
UNIFIED_MEMORY_GB = 128  (partition 80/20/20/8)  TARGET_FRAMES_PER_SECOND = 150_000
```

Derived helpers:

- `total_temporal_frames()` = `TIMELINE_YEARS·365.25·86400 // 24` ≈ **1.349 × 10¹⁰**
  frames.
- `raw_uncompressed_bytes()` = `frames · 70 · 2` ≈ **1.9 TB** (BF16, before delta
  compression; the ~300 GB target adds entropy coding).
- `memory_partition_gb()` → `{mps_tensors:80, ring_buffer:20, testing_daemon:20,
  system_overhead:8}`.
- `timeline_bounds()` → `TimelineBounds(start_jd=588465.5, end_jd=start+10256·365.25,
  n_frames)`, with `.span_days`.

---

## 2. `geometry.py` — backend‑neutral primitives

Shared numpy math used by projection, losses, the indexer, kundali, and the
decoupled engine. All angle‑aware and vectorized.

| Function | Purpose |
|---|---|
| `wrap_angle(θ)` | wrap radians to `(-π, π]` |
| `angular_separation(a, b)` | shortest angular gap between two angles |
| `to_unit_vector(lon, lat)` | `(lon,lat)`→ 3‑D unit vector (rows) |
| `geodesic_distance(u, v, eps=1e-7)` | clamped‑arccos great‑circle distance between unit vectors |
| `pairwise_angular_matrix(lons)` | `(B,B)` matrix of pairwise separations — rotation‑invariant aspect signature |
| `obliquity_of_ecliptic(jd)` | mean obliquity ε(jd) |
| `greenwich_mean_sidereal_time_deg(jd_ut)` | GMST in degrees |
| `ecliptic_to_equatorial(lon, lat, ε)` | rotate ecliptic → equatorial |

`TWO_PI` is exported. These are the correctness oracle for any GPU reimplementation.

---

## 3. `ephemeris/bodies.py` — the ten entities of `G(t)`

`ENTITIES` is the canonical, order‑stable table (index = row in `G(t)`, 0..9):

| idx | name | Kind | swe id | note |
|---|---|---|---|---|
| 0 | Sun | BODY | 0 | |
| 1 | Moon | BODY | 1 | |
| 2 | Mercury | BODY | 2 | |
| 3 | Venus | BODY | 3 | |
| 4 | Mars | BODY | 4 | |
| 5 | Jupiter | BODY | 5 | |
| 6 | Saturn | BODY | 6 | |
| 7 | Rahu | NODE | 11 (true node) | mean/true north node |
| 8 | Ketu | NODE | 11 + 180° offset | south node = Rahu + 180° |
| 9 | Ayanamsha | PRECESSION | — | precession scalar, not a body |

`Kind ∈ {BODY, NODE, PRECESSION}` decides how coordinates are obtained.
`CelestialEntity(name, kind, swe_id, longitude_offset_deg)`. `NAMES`, `index_of(name)`.

> The **decoupled engine uses a different ten** — Sun…Pluto (swe ids 0–9, adds
> Uranus/Neptune/Pluto, drops the nodes/Ayanamsha). See
> [09 §2](09-decoupled-engine.md). The original `G(t)` table is the one above.

---

## 4. `ephemeris/global_state.py` — computing `G(t)`

The core Phase‑1 module. Manages the ephemeris backend and produces the global
state matrix.

**Backend management**

- `configure(mode, ephe_path=None, jpl_file=None)` — `mode ∈ {"moshier","swiss","jpl"}`;
  sets swe ephemeris/JPL paths.
- `_calc_flags()` — returns the swe flag mask for the active backend **plus the speed
  flag** (so velocities are always returned): `moshier=4 | swiss=2 | jpl=1`, OR’d with
  `_FLG_SPEED=256`.
- `configure_from_args(ephe_path, jpl_file)` — explicit flags win, else `auto_configure()`.
- `auto_configure()` — reads a saved config from `$KALACHAKRA_CONFIG`, `./.kalachakra.json`,
  or `~/.config/kalachakra/config.json`; falls back to Moshier.
- `save_config(...)` — persist the backend choice (written by `setup_full_span.py`).
- `ephemeris_available()` / `_require_swe()` — presence guard for `pyswisseph`.

**Encoding** — `encode_body(λ, β, r, λ̇, β̇, ṙ)` packs one body's raw ecliptic state
(radians / AU / per‑day rates) into its boundary‑free 7‑vector:

```
v_i = [ cosλ·cosβ,  sinλ·cosβ,  sinβ,  λ̇,  β̇,  r,  ṙ ]
```

`_raw_state(entity, jd)` fetches the raw `(λ, β, r, λ̇, β̇, ṙ)`:
- BODY/NODE → `swe.calc_ut(jd, swe_id, flags)` → `[lon, lat, dist, lon_sp, lat_sp, dist_sp]`,
  with Ketu's +180° longitude offset applied.
- PRECESSION (Ayanamsha) → `swe.get_ayanamsa_ut(jd)` as a point on the ecliptic
  circle (β=0, r=1) with a finite‑difference rate.

**Public API**

- `global_state_frame(jd)` → `(10, 7)` for one instant.
- `global_state_batch(jds)` → `(len, 10, 7)` (loops on CPU; pyswisseph is scalar).
- `ecliptic_longitudes(jd)` → `(10,)` radians (convenience for the weather engine).

---

## 5. `ephemeris/calendar.py` — civil time ↔ Julian Day

Proleptic Gregorian, UTC, with **astronomical year numbering** (year 0 = 1 BCE).

| Function | Signature |
|---|---|
| `gregorian_to_jd` | `(year, month, day, hour=12, minute=0, second=0) → jd` |
| `jd_to_gregorian` | `jd → (year, month, day, hour, minute, second)` |
| `datetime_to_jd` | `datetime → jd` |
| `parse_datetime` | `"YYYY-MM-DD[THH:MM[:SS]][Z]" | "now" → jd` |
| `format_jd` | `jd → "YYYY-MM-DD HH:MM:SS.s UTC CE/BCE"` |

**JD conventions used across the project:** integer JD = **noon UT** (`.0`), `.5` =
midnight. The Kundali daily DB samples integer JDs (noon UT). The decoupled UI
scrubs in raw JD floats.

---

## 6. `ephemeris/timeline.py` — the temporal axis

Enumerates *when* samples are taken; touches no ephemeris.

```python
JD_STEP = VIGHATIKA_SECONDS / SECONDS_PER_DAY            # 24 s as a fraction of a day
frame_to_jd(frame_index) = KALI_YUGA_EPOCH_JD + frame_index · JD_STEP
jd_to_frame(jd)          = round((jd - epoch)/JD_STEP)
iter_chunk_ranges(chunk_frames) -> (start, end) half-open ranges over all frames
summary() -> {start_jd, end_jd, span_days, span_years, jd_step, vighatika_seconds, n_frames}
```

Frame 0 is the Kali‑Yuga epoch; each frame is one Vighatika later.

---

## 7. `ephemeris/se1_files.py` — Swiss `.se1` planning

Plans exactly which Swiss Ephemeris `.se1` data files cover the timeline (for the
full span). A `Block` has a filename tag per 600‑year segment; helpers:

- `all_blocks()`, `blocks_for_years(start, end)` — the segments intersecting a range.
- `filenames_for_years(start, end, prefixes=("sepl","semo",…))` — the exact file list
  (planets `sepl*`, moon `semo*`, asteroids as configured).
- `fmt_year(astro)` — format an astronomical year with era.

This drives `scripts/setup_full_span.py` (36 files, ~40 MB, DE431‑based, covering
3102 BCE→7154 CE). Tested in `tests/test_se1_files.py`.

---

## 8. Worked example — a real eclipse

The 2024‑04‑08 total solar eclipse, read straight from `G(t)` with no model:

```
Sun   19.40 Aries      Moon 19.36 Aries     Mercury 24.80 Aries R
harmonic resonance 4.47   structural tension 1.49   geometric potential 0.761
** SOLAR ECLIPSE ** Sun–Moon 0.04°, Moon–node 3.73°
```

The 0.04° Sun–Moon separation is the real new‑moon conjunction to hundredths of a
degree — an objective geometric fact from the ephemeris, produced by the weather
engine ([07 §1](07-inference-and-analysis.md)) on top of this foundation layer.
