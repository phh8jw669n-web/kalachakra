# Kalachakra Version 6 — Continuous SIREN Engine

A ground-up rebuild with **no grids, no textures, no time-steps**. The whole system is a
single continuous field:

```
(lat, lon, jd)  ─►  33-D topocentric sky  ─►  SIREN  ─►  CIE L*a*b*  ─►  sRGB pixel
     any float        11 bodies × N/E/Up      sin() net    3 numbers        on the globe
```

Because every stage is closed-form and differentiable, the **identical maths runs in three
places** — Python (training), JavaScript (the HUD), and GLSL (the globe). The trained
network is exported to a tiny JSON of weights and re-executed *per pixel* inside a fragment
shader, so the globe has literally infinite resolution: zoom from the whole Earth down to a
single street and the colour is always evaluated exactly, never interpolated from a texture.

The SIREN's output is a **bounded** L\*a\*b\* head — `L* = 50 + 50·tanh(zL/50)` and
`a*,b* = 90·tanh(z/90)` — so every colour is guaranteed displayable (L\* in (0,100), a\*/b\*
in ±90). The tanh is slope-1 near the centre, so the isometric metric is preserved for
typical colours and only the rare extremes are softly compressed. At the very end the shader
applies a **hue- and luminance-preserving gamut compression** so out-of-gamut resonance
spikes glow toward white organically instead of hard-clipping to a neon primary.

**Performance.** The celestial ephemeris depends only on *time*, not on the observer, so the
full Kepler/lunar solve for all 11 bodies is computed **once per frame on the CPU** and
handed to the shader as a tiny uniform array; the fragment shader then does only the cheap
per-pixel horizontal projection + SIREN. The globe also renders **on demand** — when the
clock is paused and the camera is still, the GPU idles completely. Both are exact refactors:
the maths is identical (verified to float32 against the reference), just without the
per-pixel redundancy.

| Module | What it is | Where it lives |
| --- | --- | --- |
| **1 · Core Engine** | ephemeris + SIREN + isometric training | `ephemeris.py`, `siren.py`, `losses.py`, `dataset.py`, `training.py` |
| **2 · Global Canvas** | Three.js sphere; the full pipeline in one fragment shader | `web/shader6.js`, `web/main.js` |
| **3 · Temporal Helm** | 64-bit Julian-Date clock, playback, scrubber, date jump | `web/main.js`, `web/index.html` |
| **4 · Observer's HUD** | click a point → live 33-D matrix + colour readout | `web/ephemeris6.js`, `web/siren6.js`, `web/main.js` |

---

## 0 · Prerequisites

- **Python ≥ 3.11** with PyTorch (`pip install -e '.[train]'` from the repo root, or just
  `pip install torch numpy`). `pyswisseph` is optional — used only by the test suite to
  cross-check the ephemeris; the engine itself needs no ephemeris data files.
- A **WebGL2** browser (any current Chrome / Firefox / Safari / Edge). The shader uses
  `texelFetch` and sized arrays, which require WebGL2 — Three.js selects it automatically.
- Any static file server for the `web/` folder (ES-module `import` and `fetch` do not work
  over `file://`). Python's built-in `http.server` is enough.
- **Node** is optional — only for the parity checks in §6.

All commands below are run **from the repository root** unless stated otherwise.

---

## 1 · Train the SIREN (Module 1)

Every training step draws a *fresh* random batch of floating-point `(lat, lon, jd)` triples,
turns each into its 33-D topocentric sky tensor, and trains the SIREN so that **distances in
colour space mirror distances in physical sky space** (the isometric loss). Nothing is ever
cached or gridded, so the network cannot learn artificial spatial or temporal seams.

```bash
# A solid default run (~40k steps). Add --export to write the shader weights in one shot.
python -m version6.train --steps 40000 --batch 2048 \
    --export version6/web/weights.json
```

Useful flags (`python -m version6.train --help` for the full list):

| Flag | Default | Effect |
| --- | --- | --- |
| `--steps` | `40000` | training steps. **Give it real steps** — the SIREN output layer starts near zero, so short runs look washed out. A few thousand is a smoke test; tens of thousands is a real field. |
| `--batch` | `2048` | random skies per step. Larger batches give the pairwise-distance loss more signal. |
| `--hidden` / `--hidden-layers` | `48` / `2` | network size. Bigger ⇒ more spatial detail (and a larger shader weight texture). |
| `--omega0` | `30.0` | SIREN frequency. Higher ⇒ finer, busier colour structure. |
| `--color-scale` | `20.0` | target `‖ΔLab‖ = color_scale · ‖ΔSky‖`. Larger ⇒ more vivid globe. |
| `--lr` | `1e-4` | learning rate (SIRENs prefer a small one). |
| `--out-dir` | `version6/checkpoints` | where `step_*.pt` and `model_final.pt` are written. |
| `--resume PATH` | – | continue from a checkpoint. |

Watch the log line: `loss` should fall while `L*`, `a*±`, `b*±` (the colour spread) grow —
that is the field learning to use the gamut. Checkpoints land in `version6/checkpoints/`.

The run is CPU-friendly for small nets but uses CUDA or Apple MPS automatically when present
(`--device cpu|cuda|mps` to force one).

---

## 2 · Export the shader weights (Module 1 → 2)

If you did **not** pass `--export` during training, export a checkpoint now:

```bash
python -m version6.export_weights \
    --checkpoint version6/checkpoints/model_final.pt \
    --out version6/web/weights.json
```

This writes two files next to the web client:

- **`web/weights.json`** — every layer's weights/biases, `omega0`, the architecture, the
  `color_scale`, and the bounded-head constants (`output_activation`, `lab_center`,
  `lab_lspan`, `lab_ab`). No display gauge is needed: the bounded L\*a\*b\* head already emits
  displayable colour centred near L\*=60 by the anchor, and the shader/HUD re-run the weights
  verbatim.
- **`web/golden.json`** — a handful of `(lat, lon, jd)` points with their 33-D tensor and the
  network's L\*a\*b\* output, used to verify the JS/GLSL ports (see §6).

> The web client needs `web/weights.json` to render the **trained** field. Without it, the
> page still loads but falls back to an *untrained* random SIREN and shows a notice — handy
> for checking the plumbing, useless as a result. Always export before a real viewing.

---

## 3 · Run the Global Canvas (Modules 2 – 4)

Serve the `web/` folder and open it in a browser:

```bash
cd version6/web
python -m http.server 8080
# then open http://localhost:8080/  in a WebGL2 browser
```

You should see the SIREN globe. The SIREN colour field runs in the fragment shader, so
panning and zooming re-evaluate every visible pixel exactly (the ephemeris is refreshed once
per frame on the CPU — see **Performance** above).

### Module 2 — navigating the globe
- **Drag** to orbit, **scroll / pinch** to zoom. Damping gives it inertia.
- Zoom range runs from the full globe down to street level (`minDistance` ≈ surface).

### Module 3 — the Temporal Helm (bottom bar)
- **▶ Play / ⏸ Pause** — advance the clock in real time.
- **Speed** slider — a logarithmic multiplier from `1×` up to millions×, and **negative to
  rewind**. `1×` is real-time; drag right to fast-forward centuries.
- **Scrubber** — a fluid (`step="any"`) drag across the full ±5000-year span; the label shows
  the target date as you move.
- **Jump to** — type an exact UTC date-time and press **Go** to teleport the clock there.
- **◉ Now** — snap back to the current instant.

Time is a 64-bit Julian Date. To keep GLSL's 32-bit floats precise across ten millennia, the
date is split into two uniforms — `u_baseDays` (the exact integer day count, constant during
animation) plus `u_timeOffset` (the small animated remainder) — so playback and scrubbing stay
smooth. (Far from the present, the Moon's fast motion can show faint fp32 stepping; this is an
inherent, documented limit of doing everything on-GPU.)

### Module 4 — the Observer's HUD (lower-left glass panel)
- **Click / tap** anywhere on the globe. A reticle pins that exact floating-point `(lat, lon)`.
- The panel then streams that point's live **33-D local-sky matrix** — 11 bodies, each a
  `(North, East, Up)` unit vector — recomputed by the JavaScript port of the ephemeris.
- Below it, the JS port of the SIREN shows the raw **L\*a\*b\***, the **HEX/RGB** swatch, and
  the pinned coordinates. Collapse the panel with the ▾ button in its header.

Because the HUD's JS ephemeris + SIREN are transcriptions of the very same maths the shader
runs, the swatch in the panel matches the pixel under the reticle.

---

## 4 · Tuning the look

- **More vivid** → retrain with a larger `--color-scale` (e.g. `40`). Vividness is now set at
  train time and stays inside the gamut (the bounded head + soft compression handle the rest);
  there is no view-time exposure hack to blow colours out.
- **Finer detail** → larger `--hidden` / `--hidden-layers`, or a higher `--omega0`.
- **Wider / narrower colour box** → the bounded-head constants (`lab_center`, `lab_lspan`,
  `lab_ab`) live in `version6/config.py::SirenConfig` and are exported into `weights.json`, so
  the JS/GLSL ports pick them up automatically.
- **Different time window** → `--jd-start` / `--jd-end` narrow or widen the training span; keep
  the web client's `JD_MIN/JD_MAX` in `main.js` consistent if you change it drastically.

---

## 5 · How it fits together (the one-ephemeris principle)

The ephemeris is deliberately **dependency-free**: closed-form JPL Keplerian elements
(Standish), a short lunar series, and a linear node — no `pyswisseph`, no `.se1`/`.bsp` files.
That is the whole point. The SIREN is trained on *exactly* the geometry the shader will later
compute, so a data-driven almanac (which cannot run in GLSL) would break the mirror. The price
is arc-minute-class accuracy near the present, degrading smoothly over millennia — which is all
a continuous geometric field needs. `ephemeris.py`, `web/ephemeris6.js` and the GLSL block in
`web/shader6.js` are line-for-line transcriptions of one another.

Pipeline per body: heliocentric Kepler elements → geocentric ecliptic → equatorial (obliquity)
→ local horizontal (hour angle + latitude) → Cartesian `(North, East, Up)`.

---

## 6 · Verifying correctness

**Python engine + end-to-end pipeline:**

```bash
python -m pytest tests/test_version6.py -q
```

This checks the 33-D geometry and unit vectors, batch-vs-per-sample consistency, GMST, an
**altitude cross-check against pyswisseph** near the present (skipped if `pyswisseph` is
absent), the SIREN export→re-run parity, the isometric loss and gauge anchor, the stochastic
generator, and a tiny train → checkpoint → resume → weight/golden export cycle.

**Cross-language parity (optional, needs Node):** after an export exists, the JS ports
reproduce the Python engine — the ephemeris matches to float32 exactly and the bounded SIREN
colour to ~1e-6, which is why the HUD swatch equals the shader pixel. The `web/golden.json`
file is the shared reference for these checks. The GLSL fragment (bounded head + soft gamut
compression included) has been verified to reproduce the JS pipeline exactly (0/255 per
channel) in a real WebGL2 context.

---

## 7 · Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| "no weights.json — showing an UNTRAINED SIREN" notice | You opened the page before exporting. Do §1–§2, then reload. |
| Blank page / WebGL error in the console | Browser lacks WebGL2. Use a current desktop browser. |
| `import` or `fetch` errors in the console | You opened `index.html` over `file://`. Serve `web/` over HTTP (§3). |
| Globe looks flat grey / washed out | The run was too short — the bounded head starts at neutral L\*=50 and only spreads as it learns. Train for tens of thousands of steps and re-export. |
| Colours too dull / too intense | Retrain with a different `--color-scale` (vividness is set at train time and stays in-gamut). |
| High CPU/GPU while idle | Shouldn't happen — the globe renders on demand. If it does, a control or extension may be forcing continuous redraws; check the console. |

---

## 8 · File map

```
version6/
  ephemeris.py        Module 1 — analytic topocentric ephemeris → [N,33] tensor
  siren.py            Module 1 — the SIREN network + weight export
  losses.py           Module 1 — isometric distance-matching loss + gauge anchor
  dataset.py          Module 1 — stochastic (lat,lon,jd) generator (no static data)
  config.py           dataclasses: SirenConfig / DataConfig / TrainConfig / V6Config
  training.py         training loop, checkpoints, export_weights_json()
  train.py            CLI: python -m version6.train
  export_weights.py   CLI: python -m version6.export_weights  (+ golden.json)
  web/
    index.html        page shell (top clock, HUD, Temporal Helm)
    style.css         glass-panel styling
    main.js           Modules 2–4: Three.js scene, clock, raycaster, HUD
    shader6.js        Module 2: the full ephemeris + SIREN + Lab→sRGB fragment shader
    ephemeris6.js     Module 4: JS port of ephemeris.py
    siren6.js         Module 4: JS port of the SIREN forward pass + Lab→sRGB
    weights.json      (generated) exported SIREN weights + lab_offset
    golden.json       (generated) parity reference points
```

That's the whole loop: **train → export → serve → explore.**
