# Kalachakra Version 8 — 88-D Relational SIREN Engine

Version 8 adds **relational awareness** to the continuous colour field and is **fully
self-contained** (no imports from other version folders — a lesson from v7's stale-config
break). Train it, export it, serve `version8/web/` on its own.

```
(lat, lon, jd) ─► 33-D local + 55-D chords = 88-D ─► SIREN(4×128) ─► CIE L*a*b* ─► sRGB
   any float      North/East/Zenith   pairwise dot     sin(ω0·x)     gamut-bounded    per vertex
```

| Module | What it is | Where it lives |
| --- | --- | --- |
| **1 · Backend** | 88-D state, balanced isometric loss, 4×128 SIREN, export | `state.py`, `siren.py`, `losses.py`, `dataset.py`, `training.py`, `train.py`, `export.py` |
| **2 · GPU render** | the whole 88-D net per **vertex**, interpolated across triangles | `web/shader8.js`, `web/main.js` |
| **3 · UI / Helm** | LIVE/Time-Machine clock, orbiting bodies, world map, controls | `web/main.js`, `web/planets8.js`, `web/geo.js`, `web/timecal.js` |

---

## 1 · The 88-D state (what makes v8 different)

* **33-D local grounding** — the 11 bodies' topocentric horizontal unit vectors
  `(North, East, Zenith)`: local angularity, rising/setting, culmination.
* **55-D geometric chords** — the pairwise dot products `v_i·v_j` for all C(11,2)=55 body
  pairs: mutual angular separations (conjunction ≈ +1, opposition ≈ −1) handed to the network
  explicitly, so it need not learn cross-products internally.
* Concatenated → an **88-D** flat tensor. (`version8/state.py`, mirrored in `web/state8.js`.)

The **balanced isometric loss** RMS-normalises the two halves so the 55 chords can't dominate
the 33 local vectors:
`d_sky = 0.5·‖ΔV‖/√33 + 0.5·‖ΔC‖/√55`, and the colour must be isometric to it:
`MSE(‖ΔLab‖, γ·d_sky)` with `γ=15`. (Taken over all pairs in the batch via `cdist`.)

---

## 0 · Prerequisites

- **Python ≥ 3.11** with PyTorch (`pip install torch numpy`). `pyswisseph` is optional (tests
  cross-check the ephemeris only).
- A **WebGL2** browser.
- A static server. `version8/web/` is self-contained, so you can serve **that folder directly**.

Run commands from the repository root.

---

## 2 · Train

Each step draws a fresh random batch of continuous `(lat, lon, jd)` skies, builds the 88-D
state and trains the SIREN so colour distances mirror the balanced sky distance. The
gamut-bounded head keeps colour displayable (never pure white/black).

```bash
python -m version8.train --steps 40000 --batch 2048 --export version8/web/weights.json
```

| Flag | Default | Effect |
| --- | --- | --- |
| `--steps` | `40000` | training steps. **Use real steps** — the isometric loss grows chroma gradually; a few thousand is muted, tens of thousands are vivid. |
| `--batch` | `2048` | random skies per step (more pairs = more signal). |
| `--hidden` / `--hidden-layers` / `--omega0` | `128` / `4` / `30` | SIREN size / frequency. |
| `--gamma` | `15.0` | colour scale `‖ΔLab‖ = γ·d_sky` (10–15 is the sweet spot). |
| `--out-dir` | `version8/checkpoints` | checkpoints + `train_v8.log`. |

Watch the log: `loss` falls while `L*`, `a*±`, `b*±` (the colour spread) grow.

---

## 3 · Export

If you didn't pass `--export`, export a checkpoint (also writes a `golden.json` parity file):

```bash
python -m version8.export --checkpoint version8/checkpoints/model_final.pt --out version8/web/weights.json
```

`weights.json` carries the SIREN weights + the gamut-head constants (`output_activation`,
`lab_l0`, `lab_lspan`, `lab_ab`); the shader and HUD re-run them verbatim.

---

## 4 · Run

`version8/web/` is self-contained, so serve it directly:

```bash
cd version8/web
python -m http.server 8080
# open http://localhost:8080/  in a WebGL2 browser
```

You'll see the SIREN colour field on the globe over a coastline world map, the 11 bodies as
3D spheres orbiting per the ephemeris, and a starfield. Without `weights.json` the page loads
an *untrained* SIREN and shows a notice — train + export for the real field.

### GPU model (why it's fast)
The CPU computes the 11 Earth-fixed body directions + GMST once per frame. The **vertex
shader** then, per vertex on a `SphereGeometry(128,128)` (~16k points): builds a
singularity-free local basis straight from the surface normal
(`û=normalize(P); n̂=normalize((0,1,0)−û·u_y); ê=û×n̂`), projects the bodies to the 33 local
vectors, computes the 55 chords, runs the full 4×128 SIREN, and converts the gamut-bounded
L*a*b* to sRGB. The fragment shader just interpolates that colour — 60–120 fps for a network
that would be unrenderable per pixel.

### Controls (collapsible panel)
- **Enter Time Machine / Go Live**, **Play/Pause**, **◉ Now**.
- **Clock zone** — Local / UTC / world offsets (display only).
- **Speed** — logarithmic multiplier up to millions×, negative to rewind (default 10000× so
  Play visibly animates; slide to 1× for real-time).
- **Step / tick** presets (24 s … 1 yr) + **custom step (h)** with ← / → buttons.
- **Scrubber** across the ±5000-year timeline, and **Jump to** an exact date-time.
- **Zoom** ＋/－ (also ↑/↓, scroll), **Field opacity**, **Render** (per-vertex tessellation:
  96² / 128² / 192²).
- **Overlays** — map, coastlines, graticule, planets (orbiting spheres), labels.
- **Observer HUD** — click the globe to pin a point; it streams the 33-D local matrix and the
  L*a*b*/HEX colour (computed by the JS port, matching the on-globe pixel).

**Keyboard:** `Space` play · `L` now · `←/→` step · `↑/↓` zoom · `G` grid · `R` reset.

---

## 5 · Verifying

```bash
python -m pytest tests/test_version8.py -q
```

Checks the 88-D geometry (33 local unit vectors + 55 chords), the gamut-bounded head
(L* in [5,95], a*/b* in [−80,80]), the export→re-run parity (the JS/GLSL contract), the
balanced loss, an altitude cross-check vs pyswisseph, and a train→export cycle. The browser
ports (`state8.js`, `siren8.js`, `shader8.js`) reproduce the Python engine to ~1e-6, and the
vertex-shader local basis matches `topocentric_tensor` to 5.96e-8.

---

## 6 · File map

```
version8/
  ephemeris.py     analytic topocentric ephemeris (self-contained copy)
  state.py         88-D state: 33 local + 55 chords
  siren.py         4×128 SIREN + gamut-bounded head (L=5+90·sigmoid, a/b=80·tanh)
  losses.py        balanced isometric loss (0.5 local + 0.5 chord)
  config.py        SirenConfig / DataConfig / TrainConfig / V8Config
  dataset.py       stochastic (lat,lon,jd) -> 88-D generator
  training.py      training loop, checkpoints, export_weights_json()
  train.py         CLI: python -m version8.train
  export.py        CLI: python -m version8.export  (+ golden.json)
  web/
    index.html     page shell (clock, control panel, Observer HUD)
    style.css      glass-panel styling
    main.js        scene, helm, HUD, orbiting bodies, per-vertex globe
    shader8.js     Module 2: the 88-D vertex-shader SIREN
    state8.js      JS 88-D state builder (parity with state.py)
    siren8.js      JS SIREN forward + gamut head + Lab->sRGB
    ephemeris8.js  JS ephemeris (equatorialDirs / gmstRad / topocentricTensor)
    planets8.js    the 11 bodies as 3D spheres orbiting the globe
    geo.js         ocean + Natural-Earth coastlines + graticule (+ optional earth.jpg)
    timecal.js     Julian Date <-> calendar + display timezone
    coastlines.json  the world map
    weights.json   (generated) SIREN weights
    golden.json    (generated) parity reference points
```

Train → export → serve `version8/web/` → explore.
