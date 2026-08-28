# Kalachakra Version 9 — The Topocentric Self-Attention Relational Engine

Version 9 renders a **live "energy signature" of the planets on Earth, from the observer's
point of view**. The 11 celestial bodies are treated as **tokens**; a lightweight
**self-attention** block computes their interactions *as seen from each observer's horizon*,
so the colour field varies sharply across the globe instead of being one global smear. It is
**fully self-contained** (no imports from other version folders — a lesson from v7's stale
break). Train it, export it, serve `version9/web/` on its own.

```
(lat, lon, jd) ─► 11 tokens ×[N,E,Zenith] ─► self-attention ─► pooled read-out ─► CIE L*a*b* ─► sRGB
   any float        topocentric, per-observer   Q·Kᵀ relations   per-body energy   gamut-bounded   per vertex
```

| Module | What it is | Where it lives |
| --- | --- | --- |
| **1 · Backend** | 11-token state, micro self-attention, observer-dependent isometric loss, export | `state.py`, `attention.py`, `losses.py`, `dataset.py`, `training.py`, `train.py`, `export.py` |
| **2 · GPU render** | the whole attention net per **vertex**, interpolated across triangles | `web/shader9.js`, `web/main.js` |
| **3 · UI / Helm** | LIVE/Time-Machine clock, orbiting bodies, world map, controls, energy HUD | `web/main.js`, `web/planets9.js`, `web/geo.js`, `web/timecal.js` |

---

## 1 · Why attention, and how v9 differs from v8

v8 fed the network 55 **chords** `v_i·v_j` (pairwise dot products of the bodies' topocentric
vectors). A dot product is **rotation-invariant**, so those chords are *identical for every
observer on Earth at a fixed instant* — they carry **zero** geographic signal (their std
across the whole globe is ~`3e-6`). That is the "global smear": the macro-configuration looked
the same everywhere.

v9 fixes this at two levels:

* **Architecture — learned attention, not fixed cosines.** Each body is a token
  `[North, East, Zenith]` plus a learned **body-identity** embedding (so the Sun and Pluto are
  distinguishable). A single-head block computes `softmax(Q·Kᵀ/√d + visibility_bias)·V`. The
  learned bilinear form `Q·Kᵀ = tᵢᵀ(WqᵀWk)tⱼ` is **not** rotation-invariant (unlike a plain
  dot product), so — unlike v8's chords — the relations genuinely **vary with the observer's
  horizon**. The softmax also *sharpens* near alignments/horizon crossings, giving localized
  "patches" a smooth cosine input cannot.

* **A horizon-visibility prior.** Every attention score (and pooling weight) for key body *j*
  gets `+ vis_bias · zenithⱼ`, so **above-horizon bodies dominate and below-horizon ones are
  suppressed** — literally *"a conjunction overhead spikes; the same conjunction underfoot
  zeroes out."* This is a fixed structural prior (the isometric objective alone does not reward
  peaky attention, so the domain physics is baked in rather than hoped for); the learned `Q·Kᵀ`
  content term modulates on top. With it, the attention becomes **7× more observer-dependent**
  and the pooled per-body "energy" read-out correlates ~0.75 with each body's altitude.

The final **learned-query pooling** reads the 11 tokens into one vector — its weights are the
per-body **energy contribution** shown in the Observer HUD.

### The loss — a deliberate, documented deviation from the PRD

The v9 PRD proposes training colour distance to match *the network's own attention-matrix
delta*. That is **circular and collapses**: the optimiser can zero the loss by making every
attention map identical (mush), fighting only the anchor. Training against your own internal
representation is a classic collapse trap.

So the **attention is the architecture**, but the isometric loss targets a **fixed,
observer-dependent geometric distance** that realises the PRD's *own stated physics* robustly:

```
d_local = ||ΔV|| / √33            33-D topocentric local vectors (observer-dependent)
d_rel   = ||ΔR|| / √55            55-D HORIZON-GATED chords  R_ij = g_i·g_j·(v_i·v_j)
                                      g_b = sigmoid(gate_k · zenith_b)   ← the gate breaks
                                      rotation-invariance, so R varies across the globe
d_sky   = w_local·d_local + w_rel·d_rel          (defaults 0.5 / 0.5)
L       = MSE( ||ΔLab|| , γ·d_sky ) + λ·anchor   (γ = 32)
```

The gated chords carry real spatial signal (across-globe std ~`0.18` vs the raw chords'
`3e-6`), so the globe shows crisp geography *and* relational events, with no collapse risk.

---

## 0 · Prerequisites

- **Python ≥ 3.11** with PyTorch (`pip install torch numpy`). `pyswisseph` is optional (tests
  cross-check the ephemeris only).
- A **WebGL2** browser.
- A static server. `version9/web/` is self-contained, so you can serve **that folder directly**.

Run commands from the repository root.

---

## 2 · Train

Each step draws a fresh random batch of continuous `(lat, lon, jd)` observer skies, feeds the
11 topocentric body tokens through the attention net, and trains it so colour distances mirror
the observer-dependent sky distance. The gamut-bounded head keeps colour displayable.

```bash
python -m version9.train --steps 40000 --batch 2048 --export version9/web/weights.json
```

| Flag | Default | Effect |
| --- | --- | --- |
| `--d-model` / `--d-ff` / `--d-head` | `32` / `64` / `32` | attention width / FFN / output-head size. |
| `--blocks` | `2` | stacked attention+FFN blocks. |
| `--gamma` | `32.0` | colour scale `‖ΔLab‖ = γ·d_sky`. Bigger = more vivid. |
| `--w-local` / `--w-rel` | `0.5` / `0.5` | weight on the local vs horizon-gated-chord distance. |
| `--gate-k` | `8.0` | horizon-gate steepness for the relational target (bigger = sharper coastlines). |
| `--batch` | `2048` | random skies per step (more pairs = more signal). |
| `--out-dir` | `version9/checkpoints` | checkpoints + `train_v9.log`. |

Watch the log: `loss` falls while `L*`, `a*±`, `b*±` (the colour spread) grow. The
`vis_bias` attention prior is a model constant (`AttnConfig.vis_bias`, default `3.0`).

---

## 3 · Export

If you didn't pass `--export`, export a checkpoint (also writes a `golden.json` parity file):

```bash
python -m version9.export --checkpoint version9/checkpoints/model_final.pt --out version9/web/weights.json
```

`weights.json` carries the attention weights (`W_in`, `E_body`, per-block `Wq/Wk/Wv/W1/W2` +
`tau`, `q_pool`, `tau_pool`, head), the gamut-head constants, and the training scales
(`gamma`, `w_local`, `w_rel`, `gate_k`, `vis_bias`) for provenance. The shader and HUD re-run
the network verbatim; inference is a plain forward, so the loss-only scales are ignored there.

---

## 4 · Run

`version9/web/` is self-contained, so serve it directly:

```bash
cd version9/web
python -m http.server 8080
# open http://localhost:8080/  in a WebGL2 browser
```

You'll see the attention colour field on the globe over a coastline world map, the 11 bodies
as 3D spheres orbiting per the ephemeris, and a starfield. Without `weights.json` the page
loads an *untrained* attention net and shows a notice — train + export for the real field.

### GPU model (why it's fast)
The CPU computes the 11 Earth-fixed body directions + GMST once per frame. The **vertex
shader** then, per vertex on a `SphereGeometry(seg,seg)`: builds a singularity-free local basis
from the surface normal (`û=normalize(P); n̂=normalize((0,1,0)−û·u_y); ê=û×n̂`), projects the
bodies to the 11 tokens, runs the **whole attention network** (embed → blocks → pool → head)
from a weight texture, and converts the gamut-bounded L*a*b* to sRGB. The fragment shader just
interpolates that colour — real-time for a network that would be unrenderable per pixel. Use
the **Render** selector (64² / 96² / 128²) to trade detail for framerate.

### Controls (collapsible panel)
- **Enter Time Machine / Go Live**, **Play/Pause**, **◉ Now**.
- **Clock zone** — Local / UTC / world offsets (display only).
- **Speed** — logarithmic multiplier up to millions×, negative to rewind (default 10000×).
- **Step / tick** presets (24 s … 1 yr) + **custom step (h)** with ← / → buttons.
- **Scrubber** across the ±5000-year timeline, and **Jump to** an exact date-time.
- **Zoom** ＋/－ (also ↑/↓, scroll), **Field opacity**, **Render** (per-vertex tessellation).
- **Overlays** — map, coastlines, graticule, planets (orbiting spheres), labels.
- **Observer HUD** — click the globe to pin a point; it streams the per-body **energy
  contribution** (the attention pooling weights; dimmed bodies are below the horizon), the
  33-D local matrix, and the L*a*b*/HEX colour (computed by the JS port, matching the pixel).

**Keyboard:** `Space` play · `L` now · `←/→` step · `↑/↓` zoom · `G` grid · `R` reset.

---

## 5 · Verifying

```bash
python -m pytest tests/test_version9.py -q
```

Checks the geometry (33 local + 55 gated chords), that the gated chords are observer-dependent
(unlike v8's raw chords), the gamut-bounded head, the export→re-run parity (the JS/GLSL
contract), that attention is observer-dependent and visibility-led, the isometric loss, an
altitude cross-check vs pyswisseph, and a train→export cycle.

The browser ports reproduce the Python engine to ~1e-5, and the **GLSL vertex shader was
verified in a real headless WebGL2 context** (compile + link + transform-feedback render)
to match the JS/PyTorch output to **~3e-7**.

---

## 6 · File map

```
version9/
  ephemeris.py     analytic topocentric ephemeris (self-contained copy)
  state.py         11 body tokens + horizon-gated chords (the loss target)
  attention.py     the micro self-attention model + gamut-bounded head
  losses.py        observer-dependent isometric loss (local + gated chord)
  config.py        AttnConfig / DataConfig / TrainConfig / V9Config
  dataset.py       stochastic (lat,lon,jd) -> 88-D target-feature generator
  training.py      training loop, checkpoints, export_weights_json()
  train.py         CLI: python -m version9.train
  export.py        CLI: python -m version9.export  (+ golden.json)
  web/
    index.html     page shell (clock, control panel, Observer HUD + energy read-out)
    style.css      glass-panel styling
    main.js        scene, helm, HUD, orbiting bodies, per-vertex globe
    shader9.js     Module 2: the per-vertex attention shader (+ packWeights)
    attn9.js       JS attention forward + gamut head + Lab->sRGB (parity with attention.py)
    state9.js      JS local vectors + gated chords (parity with state.py)
    ephemeris9.js  JS ephemeris (equatorialDirs / gmstRad / topocentricTensor)
    planets9.js    the 11 bodies as 3D spheres orbiting the globe
    geo.js         ocean + Natural-Earth coastlines + graticule (+ optional earth.jpg)
    timecal.js     Julian Date <-> calendar + display timezone
    coastlines.json  the world map
    weights.json   (generated) attention weights
    golden.json    (generated) parity reference points
```

Train → export → serve `version9/web/` → explore.
