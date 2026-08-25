# 04 · Data Pipelines

Every model consumes a streaming `IterableDataset`. There are three, one per
training regime. Modules: `data/dataset.py`, `data/curriculum.py`,
`decoupled_engine/dataset.py`.

---

## 1. `data/dataset.py` — `EphemerisStream` (mesh pipeline)

The Phase‑2/3 ingestion for the discrete VQ‑mesh models. Streams temporal windows
of the **projected local field** `E(t,s)` from the binary store via the ring buffer.

```python
@dataclass StreamConfig:
    window_frames: int = 64      # temporal length T of each sample
    window_stride: int = 32      # hop between windows
    node_subsample: int|None = None   # train on a random node subset if set
    max_prefetch: int = 3

class EphemerisStream(IterableDataset):
    __init__(store, grid, cfg, chunk_start_frames=None)
```

**Iteration** (`__iter__` → `_emit_windows`):

1. `RingBuffer` yields decoded chunks `(n, 10, 7)` in timeline order.
2. Slide a window of `T = window_frames` with stride `window_stride`.
3. For each frame in the window, `spatial.project(G, jd, grid) → (N, 10, 5)` and
   `decode_ecliptic(G) → lon (10,)`.
4. Stack → `E (T, N, 10, 5)` and `lon (T, 10)`; optional `node_subsample` random node
   subset; flatten body×feature → **`E (T, N, 50)`**.
5. Yield `(torch.float32 E, torch.float32 lon_seq)`.

**Worker sharding** (`_shard_for_worker`) splits *chunks* across DataLoader workers
(`chunk_start_frames[worker_id :: num_workers]`) so workers never overlap.

The batched shapes the v3 trainer sees: `E (B, T, N, 50)`, `lons (B, T, 10)`.

---

## 2. `data/curriculum.py` — progressive multi‑scale resolution

`CurriculumStream` is a drop‑in alternative to `EphemerisStream` for the v3 trainer
(`train_v3.py --curriculum`). It **generates `G(t)` on the fly** (via
`global_state_batch`) rather than reading a fixed‑cadence store, so it can sweep the
whole timeline at *any* temporal stride, and it changes that stride per epoch.

**The schedule** (`curriculum_phase(epoch) → CurriculumPhase`):

| Epochs | Phase | `stride_seconds` | Mode | Windows / span |
|---|---|---|---|---|
| 0–4 | solar | 24 h | continuous sweep | whole 10,256‑yr timeline |
| 5–9 | ascendant | 2 h | continuous sweep | whole timeline |
| 10–14 | navamsha | 24 min | micro‑burst | 1000 × 6‑month windows |
| 15–19 | degree | 4 min | micro‑burst | 2000 × 1‑month windows |
| 20+ | quantum | 24 s | micro‑burst | 5000 × 1‑week windows |

```python
@dataclass(frozen=True) CurriculumPhase:
    name; stride_seconds; mode ("continuous"|"microburst"); n_windows; window_span_days
    @property human_stride, human_plan
SUB_HOUR_THRESHOLD_S = 3600.0     # stride < 1h ⇒ micro-burst (the crucial constraint)
```

**The crucial constraint.** A full continuous sweep at sub‑hour strides would be
billions of frames. So any stride `< 1 h` switches to **random micro‑bursting**:
draw `n_windows` random calendar windows of `window_span_days` across the timeline
and yield frames only within them at the target stride. This exposes the model to
high‑velocity transit shears without an unbounded epoch.

**Mechanics** (`CurriculumStream(grid, StreamConfig, start_jd=None, end_jd=None,
seed=0, epoch=0)`):

- `set_epoch(e)` — the trainer calls this before each epoch to pick the rung.
- `_continuous_windows(phase, wid, nw)` — tile `[start, end]` at the stride into
  `window_frames`‑long windows (hop `window_stride`), sharded across workers by index.
- `_microburst_windows(phase, wid, nw)` — per‑epoch‑seeded anchors
  (`default_rng(seed + 1009·(epoch+1))`) drawn uniformly across the timeline; each
  anchor's span is walked at the stride; bursts sharded across workers (no overlap).
- `_emit(jds)` — `global_state_batch(jds) → G`, `spatial.project` each, stack →
  `(E (T,N,50), lon (T,10))`, identical output contract to `EphemerisStream`.

Defaults to the full timeline; `--timeline-start/--timeline-end` bound it for
backends without deep‑time coverage.

---

## 3. `decoupled_engine/dataset.py` — celestial+terrestrial slices

The decoupled engine's dataset (`CelestialTerrestrialStream`). It never reads a
store; each sample is a short run of consecutive celestial frames plus a fresh batch
of random terrestrial coordinates.

**`DataConfig`** (from `decoupled_engine/config.py`):

```python
start_jd, end_jd      # default = full timeline
temporal_len   = 3    # consecutive frames per slice (feeds the temporal loss)
stride_seconds = 3600 # spacing between the frames of a slice   (.stride_days)
points_per_frame = 1024   # random terrestrial coords per slice
samples_per_epoch = 4096  # slices before an epoch ends
seed = 0
```

**Each sample** is the triple:

```
celestial  (temporal_len, 10, 5)   wrap-continuous body state  [sinλ,cosλ,sinβ,cosβ,λ̇]
jds        (temporal_len,)         Julian Days of the slice (float64, kept on CPU)
coords     (points_per_frame, 2)   random (lat, lon) radians
```

Default collate batches to `(B, T, 10, 5)`, `(B, T)`, `(B, P, 2)`.

**Notes**

- **Area‑uniform sampling.** `features.sample_sphere_coords` draws
  `lon ~ U(-π,π)` and `sin(lat) ~ U(-1,1)` (so `lat = arcsin(u)`) — the uniform
  measure on S², not uniform‑in‑latitude (which would clump at the poles).
- **JD dtype.** `jds` stay **CPU float64** always. A modern JD (~2.46 M) in float32
  resolves to only ~0.25 day, and MPS has no float64; they are consumed only on the
  CPU (for GMST), so `move_batch` deliberately leaves them on the CPU.
- **Worker safety.** `_worker()` returns CPU tensors inside forked workers (an
  accelerator context can't cross a fork); `build_dataloader(cfg, batch_size,
  num_workers, device, epoch)` places tensors on `device` only when `num_workers==0`.
- `set_epoch(e)` reseeds for fresh slices/coords each epoch.

---

## 4. Choosing a pipeline

| Pipeline | Feeds | Source of frames | Multi‑scale? |
|---|---|---|---|
| `EphemerisStream` | v1/v2/v3 mesh models | pre‑built BF16 store via ring buffer | fixed cadence |
| `CurriculumStream` | v3 model (`--curriculum`) | on‑the‑fly `G(t)`, any stride | yes (per‑epoch schedule) |
| `CelestialTerrestrialStream` | decoupled Sky/Earth models | on‑the‑fly `G(t)` + random coords | yes (via `stride_seconds`) |

The mesh models want the projected field `E(t,s)`; the decoupled models want the raw
celestial tensor plus arbitrary coordinates — the two pipelines reflect the two
architectures' fundamentally different spatial representations.
