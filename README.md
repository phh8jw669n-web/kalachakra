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

## Quick start — real output, no data files, no training

```bash
pip install -e .          # numpy + pyswisseph (Moshier backend needs no data files)

# Real cosmic weather for a place and time, from real planetary geometry:
kalachakra reading --date now --lat 51.5 --lon -0.12

# Scan a year for real singularities (finds actual eclipses):
kalachakra scan --start 2024-01-01 --end 2025-01-01 --step-hours 24 --top 8

# Compute a real per-node field for the WebGL globe:
kalachakra map --date 2024-04-08T18:17 --nodes 8000 --out web/heatmap.json
```

Everything above runs on **real planetary positions** via the Swiss Ephemeris's
built-in Moshier backend (valid ~1900 BCE – 4650 CE, so all of history and the
present — **no external data files required**). Example — the real total solar
eclipse of 2024-04-08:

```
$ kalachakra reading --date 2024-04-08T18:17
  planetary positions (geocentric ecliptic longitude):
    Sun           19.40 Aries
    Moon          19.36 Aries
    Mercury       24.80 Aries  R      <- really was retrograde
    ...
  harmonic resonance :   4.47
  structural tension :   1.49
  geometric potential:  0.761   (stellium concentration R)
  ** SOLAR ECLIPSE proximity ** (sun-moon 0.04 deg, moon-node 3.73 deg)
    [+] Sun -Moon  conjunction   0.04 deg  (strength 1.00)
    [+] Mars-Saturn conjunction  1.41 deg  (strength 0.61)
```

Sun–Moon separation of 0.04° is the real new-moon conjunction to hundredths of a
degree. These are objective geometric quantities computed from the ephemeris —
no trained model, no text, no interpretation.

### The full ML pipeline on real data — one turn-key command

```bash
pip install -e ".[train]"   # + torch

# Trains and SAVES models. With no store present it auto-generates a real one.
python scripts/train.py
```

`train.py` with no arguments generates a real ephemeris store, builds the mesh
and STFNO autoencoder, trains in BF16 (Lion + cosine-annealing warm restarts),
prints the composite geodesic loss as it falls (~65% in the first tens of
steps), and writes checkpoints to `./checkpoints/`:

```
step_XXXXXX.pt    resumable (weights + optimizer + scheduler)
model_latest.pt   latest self-contained model (reload for inference)
model_final.pt    final self-contained model
```

Then use the trained model, or serve/visualize:

```bash
# Encode a real window into the 64-d latent manifold; emit potential/shear
python scripts/analyze.py --checkpoint checkpoints/model_final.pt \
    --date 2024-04-08T18:17 --out web/heatmap.json

# Serve the real field, then open web/index.html
python scripts/serve.py --date 2024-04-08T18:17 --nodes 8000
```

`python scripts/demo_pipeline.py` runs the whole chain (real `G(t)` → store →
ring buffer → projection → weather → real eclipse detection) in one script, and
`pytest` runs 88 tests.

### Full 10,256-year scale (DE441 + M4 Max)

The Kali-Yuga epoch (3102 BCE) and the far future (past 4650 CE) fall outside
Moshier's range, and the complete matrix is ~300 GB / ~13.4B frames trained over
~90 days. **One command sets it all up** — it downloads exactly the 36 Swiss
`.se1` files (DE431, ~40 MB, which cover the whole span), verifies it can compute
the 3102 BCE Kali Yuga epoch, and writes a config so every command uses the full
span automatically:

```bash
python scripts/setup_full_span.py     # download + verify + configure
python scripts/train.py --store data/full   # then just train (full span)
```

Prefer the raw JPL DE441 kernels? `python scripts/setup_full_span.py --jpl`.
**See [`instructions.txt`](instructions.txt)** for the complete guide, including
the manual file list / year coverage, segmented matrix generation, and full-scale
training on Apple MPS (`--nodes 122880 --hidden 128 --blocks 3 --modes 32`).

Optional extras: `.[cluster]` (hdbscan), `.[serve]` (fastapi), `.[all]`.

---

## Pipeline

| Phase | What | Entry point |
|------|------|-------------|
| 1 | Generate `G(t)` (13.4e9 frames) → BF16 `.mmap` store | `scripts/generate_ephemeris.py` |
| 2 | Ring-buffer stream + analytical projection `E(t,s)` | `kalachakra.projection`, `kalachakra.data` |
| 3 | Train STFNO autoencoder → 64-d latent → clusters | `scripts/train.py` |
| 4 | Broadcast potential/shear metrics; WebGL globe | `scripts/serve.py`, `web/index.html` |

```bash
# Phase 1 — generate a real store (any window in the Moshier range)
python scripts/generate_ephemeris.py --out data/store \
    --start-date 2024-01-01 --max-frames 4096 --chunk-frames 512

# Phase 3 — train + save models (auto-generates data/store if it is missing)
python scripts/train.py

# Phase 4 — serve the real field, then open web/index.html
python scripts/serve.py --date 2024-04-08T18:17 --nodes 8000
curl 'http://localhost:8000/potential?lat=48.85&lon=2.35'
```

---

## Repository layout

```
src/kalachakra/
  cli.py              the `kalachakra` command (reading / map / scan) — real output
  constants.py        canonical figures (timeline, units, memory budget) — audited
  geometry.py         numpy geodesic/astronomy primitives (the shared math core)
  ephemeris/          Phase 1: bodies, timeline, calendar, se1_files, G(t) + config
  grid/               geodesic Earth mesh (fibonacci lattice + icosphere)
  projection/         Phase 2: analytical G(t) -> E(t,s) spherical trig
  storage/            BF16 + delta memory-mapped store, async ring buffer
  data/               Phase 3: streaming IterableDataset
  models/             spherical conv, 1-D FNO, hierarchical autoencoder
  losses/             composite geodesic loss (numpy reference + torch)
  training/           Lion optimizer, trainer/checkpoints, testing daemon
  analysis/           weather engine (aspects/tension/eclipses), signatures,
                      HDBSCAN clustering, singularities
  serving/            broadcast engine + REST/gRPC schema
configs/default.yaml  operator knob board
scripts/              setup_full_span.py (one-command DE441 setup), generate_ephemeris.py,
                      train.py, analyze.py, serve.py, demo_pipeline.py
web/index.html        WebGL / Three.js cosmic-weather globe (loads real heatmap.json)
web/heatmap.json      a real precomputed field (2024-04-08 eclipse) for the globe
instructions.txt      full 10,256-year span (DE441) integration guide
tests/                88 tests (numpy + real-ephemeris + torch; each skips if dep absent)
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

The system produces **well-defined geometric quantities** from real ephemeris
geometry — angular separations, aspect resonance/tension, circular concentration,
temporal derivatives, latent norms, cluster labels — and nothing more. The
downstream interpretive framing in the blueprint (mapping "geopolitical
volatility" etc.) is **not** asserted here as fact; the outputs are objective
astronomy/geometry, and any meaning attached to them is the reader's.

What is real and runs today: the ephemeris, the projection, the cosmic-weather
signatures (validated against real eclipses and conjunctions), the per-node map,
the singularity scan, and a genuinely trainable autoencoder (loss verified to
decrease on real data). What is a scaling exercise on the user's own hardware:
generating the full ~300 GB / 13.4B-frame matrix over the entire 10,256-year span
(needs DE441) and running the ~90-day training cycle on an M4 Max.

## License

MIT — see `pyproject.toml`.
