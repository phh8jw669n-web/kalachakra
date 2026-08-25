# 06 · Training Mechanics

Losses, optimizers, the trainer, checkpoints, and the parallel testing daemon —
for the mesh pipeline. The decoupled engine's self‑supervised training is
summarized in §6 and detailed in [09](09-decoupled-engine.md).

Modules: `losses/{reference,geometric}.py`, `training/{optim,trainer,checkpoint,daemon}.py`.

---

## 1. The composite geodesic loss (mesh models)

No Euclidean MSE — the reconstruction target lives on spheres and cyclic axes.
`losses.reference` is the numpy oracle; `losses.geometric` is the torch version they
are tested to match. Three terms, weighted:

```python
@dataclass LossWeights:  geodesic = 1.0   spectral = 0.5   aspect = 0.5
```

### 1.1 Geodesic reconstruction — `geodesic_reconstruction_loss(recon, target, eps=1e-7)`

The field's first three per‑body channels are a **unit direction**
(`_split_field`); the loss is the **clamped‑arccos great‑circle error** between the
reconstructed and target directions on the unit sphere (not `‖·‖²`), averaged over
nodes/bodies/time.

### 1.2 Spectral harmonic — `spectral_harmonic_loss(recon_seq, target_seq, time_dim=1)`

FFT along **time**; penalizes divergence in both **amplitude and phase** of the
temporal spectrum. This makes the model match the *rhythm* of the field (24 s
micro‑rotation → millennial precession), not just per‑frame values.

### 1.3 Aspect relational invariance — `aspect_relational_invariance_loss(recon_lons, target_lons)`

Builds each frame's **pairwise angular‑separation matrix** of the ten bodies
(`geometry.pairwise_angular_matrix`) and penalizes divergence between recon and
target matrices. Because it operates on *relative* angles it is rotation‑invariant —
it scores conjunctions, squares, trines, oppositions without a hardcoded aspect
table.

### 1.4 `CompositeGeodesicLoss` (torch module)

```
forward(recon, target, *, recon_lons=None, target_lons=None)
  → (total, parts)   where total = 1.0·geodesic + 0.5·spectral + 0.5·aspect
```

`losses.reference.composite_loss(...)` is the numpy equivalent used as the oracle in
`tests/test_losses.py`.

---

## 2. `training/optim.py` — Lion + cosine warm restarts

```python
@dataclass OptimConfig:
    optimizer = "lion"   # "lion" | "adamw"
    lr = 1e-4   weight_decay = 0.01   betas (0.9, 0.99)
    restart_period = 10_000   # cosine T_0 (steps)
    restart_mult = 2          # T_mult
    min_lr = 1e-6
```

- **`Lion(Optimizer)`** — the Lion optimizer (Chen et al., 2023): sign‑of‑momentum
  updates with **decoupled weight decay**; ~3× less optimizer memory than AdamW, apt
  for the 80 GB MPS budget. `step(closure=None)`.
- `build_optimizer(params, cfg)` → Lion or AdamW fallback.
- `build_scheduler(optimizer, cfg)` → **cosine annealing with warm restarts**
  (`T_0=restart_period`, `T_mult=restart_mult`, floor `min_lr`).

---

## 3. `training/trainer.py` — the loop

```python
@dataclass TrainConfig:
    amp_dtype = "bfloat16"
    micro_checkpoint_seconds = 12·3600     # every 12 wall-clock hours
    era_checkpoint_years = 500             # every 500 simulated years
    grad_clip = 1.0   log_every = 50
```

`Trainer` coordinates model + optimizer + scheduler + `CompositeGeodesicLoss` +
checkpointing:

- `train_step(e, lons)` — one BF16 step: forward, composite loss, backward, grad clip,
  optimizer/scheduler step; returns loss parts.
- `fit(loader, max_steps=None)` — the loop; logs every `log_every`.
- `maybe_micro_checkpoint()` — save every 12 h (resumable: weights + optimizer +
  scheduler). `save_micro()`, `save_era(sim_year)` (every 500 simulated years),
  `_payload()`, `load(path)`.
- `select_device()` — MPS → CUDA → CPU.

> `train_v3.py` implements its **own** loop (self‑contained VQ training) rather than
> using this `Trainer`; `Trainer` drives the v1/v2 `SphericalAutoencoder`. See
> [11 §1](11-cli-and-configuration.md) for both.

---

## 4. `training/checkpoint.py` — self‑contained models

Portable checkpoints that reload for inference with no training rig:

- `save_model(path, model, cfg, neighbors, grid_xyz, …)` / `load_model(path, device)`
  — the continuous autoencoder.
- `save_quantized_model(...)` / `load_quantized_model(...)` — the RVQ‑quantized model
  (also stores `RVQConfig`).
- `_warn_projection(ckpt, path)` — warns if the checkpoint's `projection_version`
  disagrees with the current `PROJECTION_VERSION` (won't silently mix incompatible
  field semantics).

---

## 5. `training/daemon.py` — parallel testing daemon

A separate CPU process that scores era snapshots against known benchmarks **without
pausing training** (blueprint §5.3):

```python
@dataclass BenchmarkEvent:  name; jd; ...          # a known-JD event
@dataclass DaemonConfig:    ...
DEFAULT_BENCHMARKS  = eclipses, great conjunctions, synthetic resonance
```

- `resonance_divergence(model_encode, grid, benchmarks)` — the **Resonance
  Divergence Metric**: how far the model's latent response at each benchmark instant
  drifts from expectation.
- `TestingDaemon` — file‑watches the checkpoint dir (`_new_checkpoints`), scores each
  new era snapshot (`poll_once`), logs a record (`_log`), and runs a loop
  (`run(stop_after=None)`).

---

## 6. Decoupled‑engine training (self‑supervised)

`decoupled_engine/training.py` — a fully separate loop with a three‑term
self‑supervised objective (no labels, no reconstruction). Detailed math in
[09 §4](09-decoupled-engine.md); summary:

```python
@dataclass TrainConfig (decoupled):
    lr = 3e-4   weight_decay = 1e-2   warmup_steps = 200   max_steps = 5000
    batch_size = 8   grad_clip = 1.0
    w_geometric = 1.0   w_terrestrial = 0.5   w_temporal = 0.25
    geo_temperature = 0.1   geodesic_eps_deg = 0.5   amp = True
```

- **Geometric interference contrastive loss** — aligns the latent tension
  neighborhood to a wave‑mechanics harmonic descriptor (anti‑collapse).
- **Terrestrial smoothness loss** — penalizes the colour field's geodesic gradient,
  *relaxed near planetary culmination boundaries*.
- **Temporal continuity loss** — penalizes the second time‑difference of colour
  (steady fast transits pass; erratic spikes are punished).

Optimizer **AdamW + cosine‑annealing‑with‑warmup** (`cosine_warmup`), grad clip, AMP
(bf16 on CUDA, fp16 on MPS). `save_checkpoint` stores both models + optim/sched/step.

---

## 7. Training regimes at a glance

| Regime | Script | Model | Loss | Optimizer | Data |
|---|---|---|---|---|---|
| v1/v2 mesh | `train.py`, `train_v2.py` | `SphericalAutoencoder(V2)` | composite geodesic | Lion + warm restarts | `EphemerisStream` |
| v3 VQ mesh | `train_v3.py` | `VQAutoencoderV3` | geo+spec + `λ_vq·VQ` | Lion | `EphemerisStream` |
| v3 + curriculum | `train_v3.py --curriculum` | `VQAutoencoderV3` | same | Lion | `CurriculumStream` |
| decoupled | `train_decoupled.py` | Sky + Earth | contrastive+smooth+temporal | AdamW + warmup | `CelestialTerrestrialStream` |

All CLI flags are enumerated in [11](11-cli-and-configuration.md).
