# Kalachakra `version5.1` — Run Book

A GPU-native celestial-weather visualiser built on **Zero-Redundancy Metric Learning**.
The autoencoder is gone: a pure Transformer **encoder** maps a strictly non-redundant
**50-D physical state** to a 3D **OKLab** colour, trained by a **distance-preserving
(isometric) loss** — pairwise colour distances match pairwise physical distances — so
no bottleneck can "solar-overfit" by ignoring the outer planets. It is exported to
**ONNX** and run on the client GPU; a stateless server ships a ~1.6 KB coordinate
payload.

**The 50-D state:** 11 bodies (Sun..Pluto + True Node) × `[X, Y, Z, V]` = 44, where
`(X,Y,Z)` is the body's ecliptic direction as a **Cartesian unit vector** and
`V = tanh(v/15°·d⁻¹)`; plus the **Ascendant** and **Midheaven** as ecliptic Cartesian
unit vectors (6). Computed by pure vectorised trig (Asc/MC never via `swe.houses()` in
the batch loop). The frontend still fetches all **12** bodies for the 3D orbits + glow.

```
                         ┌──────────────────────── training (offline, once) ─────────────────────────┐
  Swiss/Moshier   ──►   Monte-Carlo sampler ──► single ephemeris query ──► 50-D Cartesian state
  ephemeris             (24-s quantum, 10,256 yr)   (12 calc_ut / step)     ([N,50] broadcast)
                                                                                   │
                        Transformer encoder (11 body + 1 observer token) ──► 3-neuron OKLab
                                                                                   │  isometric loss:
                                                                                   │  MSE(‖ΔOKLab‖, ‖Δstate‖)
                                                                                   ▼
                                                                           version5/web/model_v5.onnx
  ┌──────────────────────────── runtime (live) ───────────────────────────────────┴───────────────┐
  browser ── GET /telemetry ──► FastAPI (12 calc_ut, ~1.6 KB JSON) ──► onnxruntime-web (50-D → OKLab)
          ├─ Three.js dual-layer globe ◄── OKLab→sRGB + 12-body glow GLSL shader ◄── field texture ─┤
          └─ 12 celestial-body sprites orbit the Earth, synced to the same interpolated telemetry ──┘
```

Everything new lives under `version5/`. It **imports** the planetary math, calendar,
Transformer block and OKLab maths from the root `kalachakra` package — nothing is
duplicated.

---

## 0. Prerequisites

```bash
# from the repository root
python -m pip install --upgrade pip
pip install -e ".[version5]"          # torch + pyswisseph + onnx + fastapi + uvicorn
```

- **Python** ≥ 3.11.
- **pyswisseph** (installed above) — the physics engine.
- A browser with **WebGL2** (required) and ideally **WebGPU** (Chrome/Edge 113+, Safari
  Technology Preview) for the fastest neural inference. WebGL2-only browsers fall back
  to single-threaded wasm — still 60 FPS on screen, just a slower field refresh.
- **Deep time (BCE / far-future) needs the Swiss `.se1` files.** The default Moshier
  backend needs no files but only covers ~3000 BCE – 3000 CE. See §1.

The browser loads `three.js` and `onnxruntime-web` from a CDN, so the machine viewing
the page needs outbound internet (or vendor those two files locally — see §7).

---

## 1. (Optional) Enable the full 10,256-year span

Moshier (default, zero-config) is fine for a first run bounded to `--start 1900-01-01
--end 2100-01-01`. To train and serve the **entire** `-3101-02-18 … 7155-02-18` span you
need the Swiss ephemeris data files and must point the tools at them.

```bash
# one-time: fetch/point at the .se1 files (writes ~/.config/kalachakra/config.json)
python scripts/setup_full_span.py --help

# ...or just export the path; every version5 command below then uses it with no flags:
export KALACHAKRA_EPHE_PATH=/path/to/your/ephe     # directory holding the .se1 files
```

Both the trainer and the server accept an explicit `--ephe-path /path/to/ephe` that
overrides the environment.

---

## 2. Train the model

Each step draws one random 24-second-quantised timestamp, runs a single ephemeris query,
and builds the 50-D physical state for a batch of sphere-uniform observers. The encoder
maps each to an OKLab colour and the **isometric loss** matches the normalised pairwise
colour distances to the physical ones. Loss falling means the 3 colours preserve the
50-D geometry (no reconstruction/decoder involved).

**Quick run (no data files, bounded to Moshier's range, auto-exports the ONNX):**

```bash
python -m version5.train \
    --start 1900-01-01 --end 2100-01-01 \
    --steps 40000 --locations 2048 --workers 8 \
    --export version5/web/model_v5.onnx
```

**Full 10,256-year run (needs the Swiss files from §1):**

```bash
python -m version5.train \
    --start "-3101-02-18T00:00:00" --end "7155-02-18T00:00:00" \
    --ephe-path /path/to/your/ephe \
    --steps 200000 --locations 2048 --workers 10 \
    --out-dir version5/checkpoints \
    --export version5/web/model_v5.onnx
```

Useful flags:

| flag | meaning | default |
|---|---|---|
| `--locations` | observer locations per timestamp (the broadcast batch) | 2048 |
| `--steps` | training steps (each = one Monte-Carlo timestamp) | 40000 |
| `--workers` | parallel data-generation workers (CPU physics) | 0 |
| `--d-model / --layers / --dim-ff` | encoder size | 96 / 3 / 256 |
| `--device` | `mps` / `cuda` / `cpu` (auto if empty) | auto |
| `--amp` | enable autocast (off by default; little benefit for this small net) | off |
| `--resume PATH` | resume optimizer+scheduler+step from a checkpoint | — |
| `--export [PATH]` | export the encoder to ONNX when done (default `version5/web/model_v5.onnx`) | off |

Checkpoints land in `--out-dir` (`step_XXXXXX.pt` + `model_final.pt`) alongside
`train_v5.log`. The log prints loss plus OKLab health (`L`, `chroma`, `|a|`, `|b|`) and
raises `** COLLAPSE? **` if the colours flatline. A healthy run shows loss falling and a
non-trivial `chroma`.

> **Sizing.** The default net is ~290 K parameters. `--locations 2048` at ~120 K
> samples/s (Apple-silicon MPS) covers ~2 billion observer-samples in ~200 K steps —
> a few hours. Think in *samples*, not steps: pushing far past a loss plateau buys
> nothing.

---

## 3. Export the ONNX bottleneck (if you didn't use `--export`)

```bash
python -m version5.export_onnx \
    --checkpoint version5/checkpoints/model_final.pt \
    --out version5/web/model_v5.onnx
```

This exports the encoder (`state [N,50]` → `oklab [N,3]`) with a **dynamic batch axis**
and **constant folding**, checks the graph, and — if `onnxruntime` is installed —
verifies the ONNX output matches PyTorch to < 1e-4. It also
writes `version5/web/golden.json`, a parity vector the browser re-checks on load.

---

## 4. Start the telemetry server

```bash
# simplest (Moshier, current era):
uvicorn version5.server:app --reload

# full span / choose host+port:
python -m version5.server --ephe-path /path/to/your/ephe --host 127.0.0.1 --port 8000
```

The server is **stateless and torch-free** — it only runs `pyswisseph` and returns a
sub-2 KB JSON payload. It also serves the frontend and the `.onnx` model as static
files. Sanity-check it:

```bash
curl "http://127.0.0.1:8000/telemetry?time=2026-08-26T19:00:00Z"   # < 2 KB payload
curl  http://127.0.0.1:8000/api/info                               # backend + has_model
```

---

## 5. Open the live UI

Open **http://127.0.0.1:8000** in a WebGL2/WebGPU browser.

- **LIVE** mode (emerald clock): the globe shows the real-time energy field; telemetry
  refreshes once per second, the field morphs smoothly between updates.
- **Time Machine** (magenta clock): click **Enter Time Machine** to travel. The magenta
  clock warns you are off "now".

Controls (bottom-left panel) and keyboard:

| action | control | key |
|---|---|---|
| LIVE ⇄ Time Machine | *Enter Time Machine* / *Return to Live* | `L` → live |
| Play / Pause playback | *Play/Pause* | `Space` |
| Step size / tick | `24s · 1h · 1d · 1mo · 1yr` presets or custom hours | — |
| Step once (paused) | `←` / `→` | `←` `→` |
| Clock timezone | *Clock zone* selector (Local, UTC, IST +5:30, any offset — display only) | — |
| Zoom in / out | `＋` / `－` buttons, mouse wheel, or | `↑` / `↓` |
| Move / hide the panel | drag its header; `▾` hides, `⚙ Controls` restores (position remembered) | — |
| Field opacity | slider (drives the shader `u_opacity`) | — |
| Sub-planetary glow / body labels | checkboxes | — |
| Reset camera | — | `R` |
| Rotate | click-drag (OrbitControls, damped) | — |

The status line shows live **fps**, the inference **engine** (WebGPU or wasm, or
`analytic fallback` if no model is loaded), and the ephemeris **backend**.

> **No model yet?** The page still runs: with no `model_v5.onnx` it renders a physics
> *analytic fallback* (lightness from the Sun's altitude, hue from the Moon & Jupiter)
> and shows a notice. Train + export (§2–3) to light up the real neural field.

---

## 6. One-command quickstart (bounded, no data files)

```bash
pip install -e ".[version5]"
python -m version5.train --start 1990-01-01 --end 2035-01-01 \
    --steps 20000 --locations 2048 --workers 8 --export
uvicorn version5.server:app
# open http://127.0.0.1:8000
```

---

## 7. Acceptance checks (PRD page 10)

- **Micro-payload:** `curl .../telemetry` returns **< 2 KB** (12 bodies with RA/Dec,
  ecliptic longitude/latitude and velocity, plus GAST and obliquity) and no pixels.
- **Client math == server math:** open the browser console — on load it logs
  `JS<->server feature parity: max abs err = … PASS` (rebuilds `golden.json`'s features
  from telemetry with the JS engine). The Python side proves ONNX == PyTorch during
  export.
- **60 FPS:** the render loop is shader-only and runs at the monitor's refresh rate,
  independent of telemetry — the `fps` readout stays locked while you rotate.
- **Artifact-free horizons:** zoom into a colour boundary — the GPU bilinear-samples the
  neural field, so gradients are smooth with no mesh staircasing.
- **Terminator sync (LIVE):** the day/night colour break tracks the true geographic
  sunrise/sunset line for the current time.
- **Micro-movement (24 s):** select the `24s` step, pause, and press `→` — the field
  registers the tiny, correct shift, confirming the 24-second sensitivity.

Run the automated tests:

```bash
pip install pytest
pytest tests/test_version5.py -q
```

---

## 8. How it works (the short version)

- **Single query rule** (`version5/ephemeris.py`): a body's ecliptic state depends on
  *time only*, so each timestamp costs exactly **12 `calc_ut` calls** (`FLG_SPEED` for
  velocity); the equatorial `(RA, Dec)` is derived by one vectorised obliquity rotation
  (matches pyswisseph's native output to ~1e-13°). The block is broadcast over every
  observer / pixel.
- **50-D state builder** (`version5/sky_math.py`): each body's ecliptic longitude/latitude
  → Cartesian unit vector `(X,Y,Z)` + `V=tanh(v/15)`, plus the Ascendant & Midheaven from
  Local Sidereal Time + obliquity (**no `swe.houses()` in the batch loop**; Asc/MC verified
  against `swe.houses()` to ~1e-13°). Pure tensor broadcasting into `[N,50]`. The identical
  formulas live in `version5/web/skymath.js`, verified bit-for-bit (~5e-7).
- **Model** (`version5/model.py`): a **pure encoder** — no decoder. The 50-D state is read
  as 11 body tokens (`[X,Y,Z,V]`, already bounded Cartesian → no sin/cos expansion) + 1
  observer token (Asc+MC Cartesian); the self-attention Transformer block (imported from
  `kalachakra.local_autoencoder`) pools the observer token into **3 OKLab neurons**
  (`L` sigmoid, `a,b` tanh). Optimiser: AdamW, **1,000-step warmup** → cosine to `lr_min=1e-6`.
- **Isometric loss** (`version5/losses.py`): `torch.cdist` gives the `[N,N]` pairwise
  distance matrices of the 50-D state and the 3-D colour; each is normalised by its max to
  `[0,1]` and matched by MSE. If a body moves, the physical distances change and the colour
  *must* move to compensate — no bottleneck can ignore the outer planets, and a collapsed
  (constant) colour is heavily penalised.
- **Export** (`version5/export_onnx.py`): the whole encoder, **one dynamic-batch input**
  `state [N,50]` → `oklab [N,3]`, constant folding, PyTorch↔ONNX parity check (~3e-7).
- **GPU state engine** (`version5/web/skycompute.wgsl` + `gpucompute.js`): builds the
  `[N,50]` state as a **WebGPU compute shader** — one thread per grid point; the 44
  time-only body dims are precomputed on the CPU and copied, and each thread derives its
  own Ascendant/Midheaven. Own `GPUDevice` (separate from the Three.js WebGL context),
  reads back via a staging buffer + `mapAsync`, hands the `[N,50]` array to ONNX. Verified
  bit-for-bit against the CPU `skymath.js` (0.0) and validated with `naga`; falls back to
  the CPU loop when `navigator.gpu` is absent.
- **Render** (`version5/web/main.js`): telemetry → GPU/CPU 50-D state → onnxruntime-web
  → OKLab grid texture. The Three.js outer shell's GLSL fragment shader samples the
  field, converts **OKLab→sRGB on the GPU**, and runs a **per-pixel 12-body spherical
  loop** that lights each body's sub-planetary point (the glow under which the matching
  3D body floats). All 12 bodies are rendered as 3D sprites on a **radius-2.5 celestial
  sphere**, their RA/Dec projected to Three.js Y-up coordinates
  (`x=-R cosδ sinα, y=R sinδ, z=R cosδ cosα`, evaluated at the apparent hour angle so
  each body locks over its sub-planetary point) and **interpolated between telemetry
  frames** (shortest-path angle lerp) in the `requestAnimationFrame` loop —
  so at "1 Day / tick" you watch the Moon race around the Earth while Pluto barely
  creeps, perfectly synced to the energy map below.

---

## 9. Troubleshooting

- **`no model_v5.onnx yet`** — you haven't trained/exported; the page shows the analytic
  fallback. Do §2–3.
- **BCE dates return a server error** — Moshier can't reach them; start the server with
  `--ephe-path` (Swiss files) so deep time resolves.
- **Field refresh feels slow** — you're on the wasm fallback. Use a WebGPU browser, or
  lower `GRID_W/GRID_H` at the top of `main.js`. On-screen FPS is unaffected either way.
- **Earth texture missing** — the CDN Blue-Marble was blocked; drop any equirectangular
  JPG at `version5/web/earth.jpg` (loaded first, same-origin) or accept the procedural
  ocean fallback.
- **Offline / air-gapped** — download `three.module.js`, the `three` addons, and the
  `onnxruntime-web` dist into `version5/web/` and repoint the `<script>`/import-map URLs
  in `index.html` at those local copies.
- **Apple-silicon `float64` MPS errors** — not applicable here; training keeps Julian
  Days on the CPU and never sends float64 to MPS.

---

## 10. File map

```
version5/
  ephemeris.py     single 12-call ecliptic query -> +equatorial +obliquity +GAST
  sky_math.py      builds the Zero-Redundancy 50-D state (Cartesian + Asc/MC) (numpy)
  dataset.py       infinite Monte-Carlo IterableDataset (one timestamp × N observers)
  model.py         Sky-Energy metric encoder (state -> OKLab; reuses root Transformer)
  losses.py        isometric distance-preserving loss + OKLab health (reuses oklab_stats)
  training.py      loop, checkpoints, logging (reuses select_device/cosine_warmup)
  train.py         CLI: python -m version5.train
  export_onnx.py   CLI: encoder -> model_v5.onnx (2 inputs) + golden.json parity vector
  server.py        FastAPI /telemetry micro-payload + static frontend
  web/
    index.html     HUD + import map (three.js, onnxruntime-web)
    style.css       overlay styling
    skymath.js     the JS twin of sky_math.py (50-D state) + telemetry + OKLab->sRGB
    skycompute.wgsl WebGPU compute shader: builds the 50-D state per grid point
    gpucompute.js  WebGPU pipeline (device, buffers, dispatch, mapAsync readback)
    main.js        Three.js dual-layer globe, 12-body 3D orbits, ONNX inference, UI
  instructions.md  this file
```
