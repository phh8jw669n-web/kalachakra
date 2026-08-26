# Kalachakra `version5` — Run Book

A complete, GPU-native rebuild of the celestial-weather visualiser. The astrophysics
is **decoupled** from rendering: a Transformer autoencoder is trained on a Monte-Carlo
random walk across the 10,256-year timeline, its 3-neuron **OKLab** bottleneck is
exported to **ONNX**, and the browser runs that neural field on the GPU while a
stateless server ships only a ~1 KB payload of the ten bodies' coordinates.

```
                         ┌──────────────────────── training (offline, once) ─────────────────────────┐
  Swiss/Moshier   ──►   Monte-Carlo sampler ──► single ephemeris query ──► vectorised horizon math
  ephemeris             (24-s quantum, 10,256 yr)   (10 calc_ut / step)     ([N,10,5] broadcast)
                                                                                   │
                                              Transformer + <OBSERVER> ──► 3-neuron OKLab ──► decoder (MSE)
                                                                                   │  export encoder only
                                                                                   ▼
                                                                           version5/web/model_v5.onnx
  ┌──────────────────────────── runtime (live) ───────────────────────────────────┴───────────────┐
  browser ── GET /telemetry ──► FastAPI (10 calc_ut, ~1 KB JSON) ──► onnxruntime-web (OKLab grid)
          └─ Three.js dual-layer globe ◄── OKLab→sRGB GLSL fragment shader ◄── OKLab field texture ─┘
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

Each step draws one random 24-second-quantised timestamp from the whole span, runs a
single ten-call ephemeris query, and reconstructs the local sky for a batch of
sphere-uniform observers — all compressed through the 3-neuron OKLab bottleneck. Loss
falling proves the three colours describe the entire local geometry.

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

This strips the training decoder, exports **only** the encoder + 3-neuron OKLab head
with a **dynamic batch axis** and **constant folding**, checks the graph, and — if
`onnxruntime` is installed — verifies the ONNX output matches PyTorch to < 1e-4. It also
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
| Field opacity | slider (drives the shader `u_opacity`) | — |
| Reset camera | — | `R` |
| Rotate / zoom | click-drag / scroll (OrbitControls, damped) | — |

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

- **Micro-payload:** `curl .../telemetry` returns **< 2 KB** and contains no pixels.
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

- **Single query rule** (`version5/ephemeris.py`): a planet's equatorial `(RA, Dec)`
  depends on *time only*, so each timestamp costs exactly 10 `calc_ut` calls; the result
  is broadcast over every observer / pixel.
- **Vectorised horizon math** (`version5/sky_math.py`): `H = GAST + λ − α`,
  `sin a = sin φ sin δ + cos φ cos δ cos H`, azimuth via a quadrant-safe `atan2`. The
  identical formulas live in `version5/web/skymath.js`, so the browser's neural input is
  bit-for-bit the server's (verified to ~6e-7).
- **Model** (`version5/model.py`): 5 angles/body → `(sin,cos)` → `d_model` →
  `<OBSERVER>` token → self-attention (imported from `kalachakra.local_autoencoder`) →
  **3 OKLab neurons** (`L` sigmoid, `a,b` tanh). A decoder reconstructs each planet's
  altitude & azimuth under MSE — training only.
- **Export** (`version5/export_onnx.py`): encoder only, dynamic batch, constant folding.
- **Render** (`version5/web/main.js`): telemetry → JS features → onnxruntime-web → OKLab
  grid texture; the Three.js outer shell's GLSL fragment shader samples the field,
  converts **OKLab→sRGB on the GPU**, and composites it over the Earth at 60 FPS.

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
  ephemeris.py     single ten-call query -> RA/Dec + GAST (reuses root pyswisseph)
  sky_math.py      vectorised spherical trig + 24-s sampler (numpy)
  dataset.py       infinite Monte-Carlo IterableDataset (one timestamp × N observers)
  model.py         Sky-Energy Autoencoder (reuses root Transformer block)
  losses.py        mass-weighted MSE + OKLab health (reuses root oklab_stats)
  training.py      loop, checkpoints, logging (reuses select_device/cosine_warmup)
  train.py         CLI: python -m version5.train
  export_onnx.py   CLI: encoder -> model_v5.onnx (+ golden.json parity vector)
  server.py        FastAPI /telemetry micro-payload + static frontend
  web/
    index.html     HUD + import map (three.js, onnxruntime-web)
    style.css       overlay styling
    skymath.js     the JS twin of sky_math.py + OKLab->sRGB (dependency-free ESM)
    main.js        Three.js dual-layer globe, ONNX inference, time-machine UI
  instructions.md  this file
```
