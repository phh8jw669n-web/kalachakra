# 11 · CLI, Scripts & Configuration

Every entry point, every script's purpose and key flags, the config dataclasses,
the operator knob board, and the ephemeris backend/environment setup.

---

## 1. The `kalachakra` CLI (`cli.py`)

Installed as the `kalachakra` command (`[project.scripts]`). Real geometric output,
**no model, no data files** (Moshier backend):

| Command | Purpose | Key flags |
|---|---|---|
| `kalachakra reading` | objective cosmic‑weather reading for a place & time | `--date --lat --lon` |
| `kalachakra map` | per‑node field for the WebGL globe | `--date --nodes --out` |
| `kalachakra scan` | scan a span for singularities (finds real eclipses) | `--start --end --step-hours --top` |

`build_parser()`, `main(argv)`, `cmd_reading/map/scan`. Example:
`kalachakra reading --date 2024-04-08T18:17 --lat 51.5 --lon -0.12`.

---

## 2. Scripts (`scripts/`)

### 2.1 Data & indexing

| Script | Purpose | Key flags |
|---|---|---|
| `generate_ephemeris.py` | Phase 1: write the BF16 `.mmap` `G(t)` store | `--out --start-date --start-frame --chunk-frames --max-frames --ephe-path --jpl-file` |
| `build_index.py` | Phase 2/3: tokenize → three‑tier Parquet index | `--out --checkpoint --quantized-checkpoint --nodes --start-date --frames --window --rarity-min --block-days` |
| `build_kundali_db.py` | build the daily sidereal DuckDB | `--start-year --end-year --out` |
| `great_indexer.py` | the Great Indexer → `dossiers.sqlite` | `--checkpoint --out-dir --start-date --end-date --days --full --lite --coarse-seconds --fine-seconds --velocity-threshold --chunk-frames --heartbeat-frames --heartbeat-seconds --calib-days --epoch-years --node-batch --device` |
| `setup_full_span.py` | download/verify/configure the full‑span backend | `--dest --start-year --end-year --mirror-all --jpl --base-url --config-path --configure-only --dry-run --force --no-verify --no-config` |

### 2.2 Training

| Script | Model | Key flags |
|---|---|---|
| `train.py` | `SphericalAutoencoder` (v1) | `--store --checkpoints --nodes --hidden --latent --modes --blocks --knn --quantize --window --stride --batch --epochs --max-steps --lr --resume` |
| `train_v2.py` | `SphericalAutoencoderV2` (node‑chunked) | v1 flags + chunking knobs |
| `train_v3.py` | `VQAutoencoderV3` | `--store --nodes --window --stride --batch --hidden --blocks --modes --codebook --beta --lambda-vq --node-chunk --vq-chunk --grad-checkpoint --curriculum --curriculum-seed --timeline-start --timeline-end --save-every --log-every --checkpoints --empty-cache-every` |
| `train_decoupled.py` | Sky + Earth | see [09 §8](09-decoupled-engine.md) — data/model/optim/loss‑weight/io flags |

### 2.3 Inference & serving

| Script | Purpose | Key flags |
|---|---|---|
| `analyze.py` | load a trained model → latent energy signatures / heatmap | `--checkpoint --date --window --out --top --sigma` |
| `profile_archetypes.py` | "Absolute Weather Map" archetype profiler | `--checkpoint --out --days --years --batch --seed` |
| `demo_pipeline.py` | end‑to‑end smoke run on real data | (none) |
| `serve.py` | Cosmic Weather Broadcast API + globe | `--date --fields --nodes --frame --demo --host --port` |
| `serve_radar.py` | radar control plane + `/ui/radar.html` | `--index --host --port` |
| `serve_grpc.py` | gRPC `CosmicWeather` | `--index --host --port` |
| `serve_resonance.py` | 3‑D resonance viewer | `--host --port …` |
| `serve_kundali.py` | Kundali twin dashboard | `--db --host --port` |
| `serve_decoupled.py` | live energy‑field dashboard | `--checkpoint --host --port --device --bank-size --ephe-path --jpl-file` |

Turn‑key: `python scripts/train.py` (no args) auto‑generates a store and trains;
`python scripts/demo_pipeline.py` runs the whole chain.

---

## 3. Config dataclasses (index)

| Dataclass | Module | Governs |
|---|---|---|
| `AutoencoderConfig` | `models.autoencoder` | v1/v2 mesh model |
| `VQAutoencoderV3Config` | `models.autoencoder_v3` | v3 VQ model |
| `RVQConfig` | `models.rvq` | residual VQ tree |
| `LossWeights` | `losses.geometric` | composite loss weights |
| `OptimConfig` | `training.optim` | Lion/AdamW + warm restarts |
| `TrainConfig` | `training.trainer` | mesh training loop |
| `StreamConfig` | `data.dataset` | mesh dataset windows |
| `CurriculumPhase` | `data.curriculum` | curriculum schedule rung |
| `IndexerConfig` | `indexer.config` | the Great Indexer |
| `EngineConfig` (+ `SkyEncoderConfig`, `EarthLensConfig`, `DataConfig`, `TrainConfig`) | `decoupled_engine.config` | the decoupled engine |

Exact defaults for each are in the relevant topic doc ([05](05-models.md),
[06](06-training.md), [07 §8](07-inference-and-analysis.md), [09 §2](09-decoupled-engine.md)).

---

## 4. `constants.py` & `configs/default.yaml`

`constants.py` is the audited source of truth (see [02 §1](02-ephemeris-and-geometry.md)).
`configs/default.yaml` mirrors it as an operator knob board — **nothing reads it
implicitly**; scripts take explicit CLI flags. It documents canonical settings:
timeline, grid (`n_nodes:122880, scheme:fibonacci`), global_state widths, storage
(`chunk_frames:1e6, dtype:bfloat16, delta_encode:true`), model
(`hidden:128, latent:64, fourier_modes:32, knn:7, n_blocks:3`), loss
(`geodesic:1.0, spectral:0.5, aspect:0.5`), optim (`lion, lr:1e-4, wd:0.01, betas
[0.9,0.99], restart_period:10000, restart_mult:2, min_lr:1e-6`), training
(`window_frames:64, window_stride:32, node_subsample:4096, batch:4, workers:4,
micro_checkpoint_hours:12, era_checkpoint_years:500, grad_clip:1.0`), analysis
(`singularity_sigma:4.0, hdbscan_min_cluster_size:50`), serving (`host, port:8000`),
hardware (`128 GB / 546 GB·s / mps:80 / 150000 fps / 90 days`).

---

## 5. Ephemeris backends & environment

`G(t)` needs a Swiss Ephemeris backend (`ephemeris.global_state`):

| Backend | Coverage | Select |
|---|---|---|
| Moshier (default) | ~3000 BCE–3000 CE, no files | default |
| Swiss (`.se1`) | full 10,256‑yr span | `--ephe-path DIR` (or `configure(mode="swiss")`) |
| JPL (DE441) | full span | `--jpl-file FILE` |

`auto_configure()` search order: `$KALACHAKRA_CONFIG` → `./.kalachakra.json` →
`~/.config/kalachakra/config.json` → Moshier. `setup_full_span.py` downloads the 36
`.se1` files (~40 MB, DE431‑based; 3102 BCE→7154 CE) and writes that config so every
command uses the full span with no flags.

**Accelerator notes.** Training auto‑selects MPS → CUDA → CPU. On Apple MPS: use
`float16` autocast (bf16 is patchy on Metal), keep Julian Days on the CPU as float64
(MPS has no float64), and use the node‑chunked v2/v3 models for the full mesh.

---

## 6. Install extras (`pyproject.toml`)

```
pip install -e .                 # numpy + pyswisseph (core; Moshier, no files)
pip install -e ".[train]"        # + torch                 (models, training)
pip install -e ".[index]"        # + pyarrow duckdb h3 pandas psutil (index/indexer)
pip install -e ".[serve]"        # + fastapi uvicorn        (REST/dashboards)
pip install -e ".[grpc]"         # + grpcio(-tools)         (gRPC)
pip install -e ".[cluster]"      # + hdbscan scikit-learn   (HDBSCAN)
pip install -e ".[transducer]"   # + scipy                  (SH topography)
pip install -e ".[all]"          # everything
pip install -e ".[dev]"          # pytest, ruff
```

`requires-python >= 3.11`. DE441 is **not** a pip package — it is Swiss `.se1` data
files, needed only for the full span.
