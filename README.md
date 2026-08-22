# Kalachakra

**An autonomous, unsupervised ML system that maps the continuous
spatio-temporal geometric waves of the solar system onto a geodesic model of
Earth — a "cosmic weather radar" that broadcasts objective mathematical energy
signatures (structural tension, harmonic resonance, geometric singularities)
instead of text.**

It computes the global ephemeris state `G(t)` over an exact 10,256-year timeline,
projects it analytically onto 122,880 observer nodes to form the local field
`E(t, s)`, compresses that field through a **Spherical Autoencoder + Spatio-
Temporal Fourier Neural Operator** into a 64-dimensional latent manifold, and
turns the latent geometry into per-coordinate potential/shear metrics served to
a WebGL globe.

See **[ARCHITECTURE.md](ARCHITECTURE.md)** for the full blueprint mapped to code.

---

## What runs today

The entire **numpy mathematical core** works with no heavyweight dependencies,
and the **PyTorch neural core** runs on CPU/CUDA/MPS:

```bash
pip install -e .                 # installs numpy only (the core)
python scripts/demo_pipeline.py  # runs the whole numpy pipeline on synthetic data
pytest                           # 46 tests: numpy core + torch smoke tests
```

`demo_pipeline.py` walks synthetic `G(t)` → BF16/delta store → ring buffer →
analytical projection → energy signatures → singularity detection → broadcast
query, so you can watch data flow through every stage without a 90-day run.

The full production pipeline needs optional extras:

```bash
pip install -e ".[ephemeris]"    # pyswisseph (Phase 1: generate G(t) from DE441)
pip install -e ".[train]"        # torch     (Phase 3: train the autoencoder)
pip install -e ".[cluster]"      # hdbscan   (Phase 3: manifold clustering)
pip install -e ".[serve]"        # fastapi   (Phase 4: broadcast API)
pip install -e ".[all]"          # everything
```

---

## Pipeline

| Phase | What | Entry point |
|------|------|-------------|
| 1 | Generate `G(t)` (13.4e9 frames) → BF16 `.mmap` store | `scripts/generate_ephemeris.py` |
| 2 | Ring-buffer stream + analytical projection `E(t,s)` | `kalachakra.projection`, `kalachakra.data` |
| 3 | Train STFNO autoencoder → 64-d latent → clusters | `scripts/train.py` |
| 4 | Broadcast potential/shear metrics; WebGL globe | `scripts/serve.py`, `web/index.html` |

```bash
# Phase 1 — first 10k frames (full run is ~300 GB)
python scripts/generate_ephemeris.py --out data/ephemeris \
    --max-frames 10000 --chunk-frames 5000 --ephe-path /path/to/de441

# Phase 3 — train
python scripts/train.py --store data/ephemeris --checkpoints checkpoints \
    --node-subsample 4096 --window 64 --batch 4 --max-steps 100000

# Phase 4 — serve (demo field, no trained model needed) then open web/index.html
python scripts/serve.py --demo --nodes 4096 --port 8000
curl 'http://localhost:8000/potential?lat=48.85&lon=2.35'
```

---

## Repository layout

```
src/kalachakra/
  constants.py        canonical figures (timeline, units, memory budget) — audited
  geometry.py         numpy geodesic/astronomy primitives (the shared math core)
  ephemeris/          Phase 1: bodies, timeline, G(t) via pyswisseph
  grid/               geodesic Earth mesh (fibonacci lattice + icosphere)
  projection/         Phase 2: analytical G(t) -> E(t,s) spherical trig
  storage/            BF16 + delta memory-mapped store, async ring buffer
  data/               Phase 3: streaming IterableDataset
  models/             spherical conv, 1-D FNO, hierarchical autoencoder
  losses/             composite geodesic loss (numpy reference + torch)
  training/           Lion optimizer, trainer/checkpoints, testing daemon
  analysis/           energy signatures, HDBSCAN clustering, singularities
  serving/            broadcast engine + REST/gRPC schema
configs/default.yaml  operator knob board
scripts/              generate_ephemeris.py, train.py, serve.py, demo_pipeline.py
web/index.html        WebGL / Three.js cosmic-weather globe
tests/                46 tests (numpy core always; torch tests skip if absent)
docs/                 per-phase notes
```

## Design invariants

- **Native units:** time in Vighatikas (24 s → ~0.1° horizon/frame); space as
  angular separation on a geodesic mesh — no 60-minute hour, no 360° grid.
- **Boundary-free encodings:** every cyclic angle enters the network as
  `(cos, sin)` / unit vectors, so there is no 0/360° discontinuity.
- **Global/local decoupling:** `G(t)` once per frame; `E(t, s)` is a pure
  broadcast projection — no per-observer physics loop.
- **Objective topology:** the system reports geometric potential and temporal
  shear, never event predictions or interpreted text.

## Notes on scope & claims

This repository implements the *architecture and mathematics* faithfully and
tests them. The downstream interpretive framing in the blueprint (mapping
"geopolitical volatility" etc.) is not asserted here as fact — the code produces
well-defined geometric quantities (latent norms, temporal derivatives, cluster
labels) derived from ephemeris geometry, and nothing more.

## License

MIT — see `pyproject.toml`.
