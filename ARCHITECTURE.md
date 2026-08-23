# Project Kalachakra — Architecture

> An autonomous, unsupervised machine-learning system that maps, compresses, and
> analyzes the continuous spatio-temporal geometric waves of the solar system
> over an exact **10,256-year** timeline, projecting the global ephemeris state
> onto a high-resolution geodesic model of Earth and broadcasting objective
> mathematical energy signatures — structural tension, harmonic resonance, and
> geometric singularities — with no linguistic lookup tables and no human bias.

Each section below states the design and points to the module that realizes it.
Terminology and figures follow the blueprint; the numbers are reproduced in code
(`kalachakra.constants`) so they can be audited rather than trusted.

---

## 1. Vision and constraints

### 1.1 Executive summary
Traditional astrological software reduces planetary physics to discrete text
lookup tables. Kalachakra discards that paradigm and treats planetary orbits as
continuous wave generators in a fluid topological matrix. It projects the global
ephemeris state onto a spherical grid of Earth and uses a **Spatio-Temporal
Fourier Neural Operator** combined with a **Spherical Autoencoder** to map wave
interference. The output is a 4-D topological simulation broadcasting real-time
mathematical energy signatures — a global "cosmic weather radar."

### 1.2 Core design philosophies
1. **Harmonic metric independence.** Time is sampled natively in *Vighatikas*
   (24 s); space is a non-Euclidean geodesic mesh where distance is angular
   separation from the planetary core. → `constants.VIGHATIKA_SECONDS`,
   `grid.geodesic`.
2. **Global/local decoupling.** The global state `G(t)` is computed once per
   frame; the local field `E(t, s)` is an analytical geometric projection of
   `G(t)` onto each observer, executed as one broadcast matrix op. →
   `ephemeris.global_state`, `projection.spatial`.
3. **Topology over narrative.** The model identifies structural wave
   interference (constructive = low-friction corridors, destructive = high
   shear) and compresses it into a latent bottleneck, clustering objective
   energy signatures rather than predicting events. → `models`, `analysis`.

### 1.3 Pipeline
```
Phase 1 (CPU)  DE441 + pyswisseph  ->  G(t) for 13.4e9 frames  ->  BF16 .mmap store
Phase 2 (GPU)  .mmap -> ring buffer RAM -> spherical-trig broadcast -> E(t,s) on 122,880 nodes
Phase 3 (GPU)  E(t,s) -> 3D/spherical encoder -> 64-d latent -> HDBSCAN clusters
Phase 4 (API)  potential & shear metrics -> gRPC/REST -> WebGL heatmap
```
→ `scripts/generate_ephemeris.py`, `scripts/train.py`, `scripts/serve.py`,
`web/index.html`.

### 1.4 Hardware budget (Apple Silicon M4 Max, zero cloud)
128 GB unified memory @ 546 GB/s, partitioned **80 / 20 / 20 / 8 GB** across the
MPS tensor pool, the async ring buffer, the parallel testing daemon, and macOS.
The blueprint's target throughput is ~150,000 frames/s (a ~90-day full cycle);
in practice wall-clock is hardware- and sampling-dependent — a uniform
sequential pass over all ~13.4B frames is far longer, so train on a stratified
sample rather than every frame (see instructions.txt Part 5/9). →
`constants.memory_partition_gb`.

---

## 2. Foundations (Phase 1)

### 2.1 Astronomical bounds
Swiss Ephemeris wraps NASA JPL **DE441** — the widest window of provably
zero-error mechanics. The timeline is anchored to the Kali Yuga epoch
**3102-02-18 BCE 00:00 UTC = JD 588465.5** and runs exactly **10,256 years** to
**7154 CE**, a closed manifold inside DE441's accuracy threshold. →
`constants.KALI_YUGA_EPOCH_JD`, `ephemeris.timeline`.

### 2.2 Native units
- **Time:** the Vighatika (24 s). At this step the eastern horizon advances
  ~0.1°/frame (`constants.HORIZON_ADVANCE_DEG_PER_FRAME`), aligning ingestion
  with Earth's rotation and avoiding aliasing.
- **Space:** a geodesic mesh of **122,880** uniform nodes (Level-5 hierarchical
  spatial index). Distance is pure angular separation. → `grid.geodesic`.

  *Implementation note:* a standard recursively-subdivided icosphere yields
  `10·4ⁿ+2` vertices, which does not land on 122,880; `grid.geodesic.default_grid`
  therefore realizes exactly 122,880 near-uniform nodes with a spherical
  Fibonacci lattice, and `icosphere()` is provided for the literal icosahedral
  construction.

### 2.3 Global state vector `G(t) ∈ ℝ^{10×7}`
Ten entities: Sun, Moon, Mercury, Venus, Mars, Jupiter, Saturn, the nodes
**Rahu**/**Ketu**, and the precession (**Ayanamsha**) vector. Each is encoded as
a smooth, boundary-free 7-vector

```
v_i(t) = [ cos λ cos β, sin λ cos β, sin β, λ̇, β̇, r, ṙ ]
```

so the network sees no 360° discontinuity. → `ephemeris.global_state.encode_body`,
`ephemeris.bodies`.

---

## 3. Spatial projection & storage (Phase 2)

### 3.1 Analytical projection `G(t) → E(t, s)`
Closed-form spherical trigonometry maps the global state onto every observer
with no per-coordinate ephemeris query: compute local sidereal time → RAMC,
then Azimuth θ, Altitude h, and the Ascendant offset Δφ, encoded as

```
e_i(s, t) = [ cos θ cos h, sin θ cos h, sin h, cos Δφ, sin Δφ ]
```

Executed as one broadcast tensor multiply across all 122,880 nodes. The numpy
reference (`projection.spatial.project`) is the correctness oracle for the Metal
kernel.

### 3.2 Memory-mapped binary storage
`G(t)` is serialized to contiguous `.mmap` chunks in **BF16** with **temporal
delta encoding**. BF16 halves the raw float32 footprint to **~1.9 TB** for the
full timeline; the delta stream is smooth and highly compressible, so adding
entropy coding (not yet enabled) is the path toward the ~300 GB target. A 20 GB async **ring
buffer** anticipates the training horizon, evicting processed chunks and
streaming upcoming epochs ahead of the GPU. → `storage.binary_store`,
`storage.ring_buffer`.

### 3.3 Ingestion pipeline
A custom `IterableDataset` for continuous streams: multi-worker readers pull
decoded chunks from the ring buffer, build `G(t)` tensors, and hand them to the
projection kernel — overlapping CPU parsing with GPU compute for zero-latency
delivery. → `data.dataset.EphemerisStream`.

---

## 4. Neural core (Phase 3)

Rejects plain CNNs/transformers in favor of a **Hierarchical Spherical
Autoencoder + Spatio-Temporal FNO** that operates natively on non-Euclidean
manifolds and continuous wave equations.

- **4.2 Encoder** — geodesic (spherical) convolutions mix space; 1-D Fourier
  layers mix time in the spectral domain (24 s micro-rotation → millennial
  precession, no phase lag). → `models.spherical_conv`, `models.fno`.
- **4.3 Latent bottleneck** — a continuous 64-dimensional code `z(t, s)`
  (`constants.LATENT_DIM`). Each code summarizes a temporal window and spatial
  neighborhood, forcing geometric invariants rather than memorized samples.
- **4.4 Decoder** — a mirror network reconstructs `E(t, s)`, providing the
  unsupervised reconstruction signal. → `models.autoencoder`.

---

## 5. Training (Phase 3)

### 5.1 Composite geodesic loss
No Euclidean MSE. Three terms (`losses.reference` = numpy oracle,
`losses.geometric` = torch):
1. **Geodesic reconstruction** — clamped-arccos great-circle error on the unit
   sphere.
2. **Spectral harmonic** — FFT along time; penalizes amplitude + phase
   divergence.
3. **Aspect relational invariance** — divergence of the multi-body pairwise
   angular-separation matrices (rotation-invariant; scores conjunctions,
   oppositions).

### 5.2 Optimization
BF16 mixed precision; **Lion** optimizer (AdamW fallback) with decoupled weight
decay and **cosine annealing with warm restarts**. → `training.optim`.

### 5.3 Checkpointing & parallel testing daemon
Micro-checkpoints every 12 h; era snapshots every 500 simulated years. A
separate CPU daemon watches for snapshots and scores each against known
benchmarks (eclipses, great conjunctions, synthetic resonance), logging a
**Resonance Divergence Metric** without pausing training. → `training.trainer`,
`training.daemon`.

---

## 6. Analysis (Phase 3)

With the decoder severed, analysis runs on `z(t, s)`:
- **6.1 Energy signatures** — Geometric Potential Field `‖z‖` (mass
  convergences) and Temporal Shear Gradient `‖dz/dt‖` (phase transitions). →
  `analysis.signatures`.
- **6.2 Manifold clustering** — HDBSCAN groups recurring interference patterns
  into objective clusters. → `analysis.clustering`.
- **6.3 Singularity detection** — robust statistical thresholding flags
  coordinates of maximum structural tension. → `analysis.anomaly`.

---

## 7. Serving (Phase 4)

- **7.1 Broadcast engine** — spatial query processor over the 122,880-node mesh.
  → `serving.broadcast`.
- **7.2 API** — gRPC/REST returning `{potential_index, shear_velocity,
  cluster_id}` JSON. → `serving.api`.
- **7.3 Interface** — WebGL/Three.js globe with vertex displacement + heat
  coloring of standing-wave interference and singularity foci. → `web/index.html`.
- **7.4 Roadmap** — storage matrix → projection kernel → STFNO training →
  HDBSCAN clustering → WebGL activation.

---

## Status

**This runs on real astronomical data today.** `G(t)` is computed from the Swiss
Ephemeris (default Moshier backend, no data files, ~3000 BCE – 3000 CE); the
`kalachakra` CLI produces real cosmic-weather readings, per-node maps, and
singularity scans, validated against real eclipses (e.g. 2024-04-08: Sun–Moon
0.04°) and conjunctions. The `analysis.weather` engine derives the objective
signatures — harmonic resonance, structural tension, geometric potential,
temporal shear, eclipse proximity, local intensity — directly from real geometry,
with **no trained model required**.

The **neural core and training loop are implemented and demonstrably learn on
real projected data** (composite geodesic loss drops ~65% in 60 steps; a real
checkpoint is saved). 88 tests cover the numpy core, the real ephemeris, and the
torch model/trainer.

What remains a **scaling exercise on the target hardware**: the full 10,256-year
span reaches outside the Moshier window (the 3102 BCE epoch and post-3000 CE need
DE441/DE431 `.se1` files), and the complete ~1.9 TB / 13.4B-frame matrix, the
sparse rarity-thresholded full-mesh index, plus the long training cycle assume
the M4 Max unified-memory pipeline of §1.4 (memory-tuned; wall-clock depends on
sampling — see instructions.txt Part 5/9).
