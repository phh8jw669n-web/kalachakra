# Kalachakra Version 7 — Regional Energy Atlas

Version 7 keeps Version 6's physics and its learned colour field **exactly**, but changes
*where* the field is evaluated so the UI runs at a smooth 60 fps.

```
v6:  every screen pixel  ->  full ephemeris + SIREN  (millions of sines / frame -> lag)
v7:  a regional grid     ->  full ephemeris + SIREN  ->  small texture  ->  mapped on a globe
        (baked off-thread, only on time change)         (a trivial texture lookup renders it)
```

Instead of an infinite per-pixel field, v7 samples a **structured, high-density grid** — a
regional lat/lon lattice plus a curated set of major metropolitan hubs — bakes those node
colours into a small equirectangular texture **on a background Web Worker**, and the globe
simply **maps that texture** onto a sphere. Rendering is then one texture lookup per pixel
(no SIREN, no Kepler, no transcendental sine sweep), so it holds 60 fps at any zoom or window
size; the field texture refreshes off the main thread as time advances.

**Reuse, no redundancy.** The topocentric ephemeris, the SIREN with its bounded /
soft-clamped L\*a\*b\* head, the isometric loss and the calendar utilities are imported
directly from Version 6 — Python for training, and the same `version6/web` ES modules for the
browser field worker and HUD. Version 7 adds only the structured dataset, the grid/city
manifest, and the texture-mapping frontend.

| Piece | What it is | File |
| --- | --- | --- |
| Structured dataset | major hubs + regional lattice, across the 10 ky timeline | `cities.py`, `dataset.py` |
| Backend training | reuses the v6 bounded-head SIREN + isometric loss | `train.py`, `training.py`, `config.py` |
| Export bundle | `weights.json` + `cities.json` + `manifest.json` | `export.py` |
| Field baker | grid → equirectangular RGBA (reuses v6 ephemeris + SIREN) | `web/field.js` |
| Background worker | bakes the field off the main thread | `web/fieldworker.js` |
| Globe | texture-mapped sphere + city markers + HUD picking | `web/globe.js` |
| App / UI | Temporal Helm, timezone, opacity, zoom, HUD, … | `web/main.js`, `web/timecal.js` |

---

## 0 · Prerequisites

- **Python ≥ 3.11** with PyTorch (`pip install torch numpy`, or `pip install -e '.[train]'`).
  `pyswisseph` is optional (used only by tests to cross-check the ephemeris).
- A **WebGL2** browser (any current Chrome / Firefox / Safari / Edge).
- A static file server. **Serve from the repository root**, because the v7 frontend imports
  the reused modules from `../../version6/web/…`. Python's `http.server` is enough.

All commands below are run **from the repository root**.

---

## 1 · Train (backend)

Every step draws a fresh structured batch — a mix of **curated metropolitan hubs**, a
**regional lat/lon lattice**, and a **uniform remainder** (so the field stays valid between
nodes) — each at a random Julian Date across the ~10,000-year timeline. The bounded,
soft-clamped L\*a\*b\* head (reused from v6) keeps every colour inside the human-perceivable
gamut (L\* in 0–100, a\*/b\* bounded), so the neon-cyan / white clipping is gone by
construction.

```bash
# solid default run; --export writes the full frontend bundle in one shot
python -m version7.train --steps 40000 --batch 2048 --export version7/web
```

Useful flags (`python -m version7.train --help` for all):

| Flag | Default | Effect |
| --- | --- | --- |
| `--steps` | `40000` | training steps. **Give it real steps** — the isometric loss grows chroma slowly; a few thousand looks muted, tens of thousands are vivid. |
| `--batch` | `2048` | nodes per step. |
| `--city-frac` / `--grid-frac` | `0.35` / `0.45` | share of each batch from hubs / lattice (the rest is uniform). |
| `--grid-step` | `5.0` | regional lattice spacing (degrees). |
| `--grid-w` / `--grid-h` | `180` / `90` | the render grid baked into `manifest.json` (the frontend default). |
| `--hidden` / `--hidden-layers` / `--omega0` | `48` / `2` / `30` | SIREN size / frequency. |
| `--color-scale` | `20.0` | vividness (`‖ΔLab‖ = color_scale·‖ΔSky‖`), safely inside the bounded gamut. |
| `--out-dir` | `version7/checkpoints` | checkpoints. |

Watch the log: `loss` falls while `L*`, `a*±`, `b*±` (the colour spread) grow.

---

## 2 · Export the frontend bundle

If you did not pass `--export`, export a checkpoint:

```bash
python -m version7.export --checkpoint version7/checkpoints/model_final.pt --out version7/web
```

This writes three files the texture-mapping frontend consumes:

- **`web/weights.json`** — the SIREN (with the bounded-head constants). The field worker
  re-runs these to colour each grid node.
- **`web/cities.json`** — the curated metropolitan hubs (`name, lat, lon, region`), drawn as
  interactive markers.
- **`web/manifest.json`** — the render grid (`width × height`), the timeline (`jd_start/end`)
  and the body list, so the frontend and backend agree.

---

## 3 · Run

Serve the **repository root** (so `../../version6/web/…` resolves) and open the v7 page:

```bash
python -m http.server 8080          # run at the repo root
# then open http://localhost:8080/version7/web/  in a WebGL2 browser
```

You should see the globe with the regional field mapped onto it and city hubs glowing. Pan /
zoom is always smooth; the field re-bakes off-thread as time advances (the `field N/s`
readout shows the bake rate; `fps` shows the render rate).

> Without `weights.json` the page still loads and shows an *untrained* field with a notice —
> useful for checking the plumbing. Train + export for the real atlas.

### Module 2/3 — the Temporal Helm (control panel, draggable / hideable)
- **Enter Time Machine / Go Live** — LIVE tracks the real clock; the Time Machine freezes,
  plays, scrubs and jumps.
- **Play / Pause**, **Now** (jump to the present).
- **Clock zone** — Local, UTC, IST, and the world offsets (display only; never affects the
  physics/UTC).
- **Speed** — a logarithmic multiplier up to millions×, negative to rewind.
- **Step / tick** presets (24 s, 1 h, 1 d, 1 mo, 1 yr) + a **custom step** (hours) with
  ← / → single-step buttons.
- **Scrubber** — a fluid drag across the whole ±5000-year timeline.
- **Jump to** — type an exact UTC date-time and press **Go**.
- **Zoom** ＋ / － buttons (also ↑ / ↓ and scroll).
- **Field opacity** — fades the field over the dark base globe.
- **Field detail** — Regional 120×60 / Balanced 180×90 / Fine 256×128 (trades update rate
  for spatial detail).
- **Overlays** — city hubs, graticule.

### Module 4 — the Observer HUD (lower-left)
- **Click / drag** anywhere on the globe to pin a float lat/lon; a reticle marks it and the
  **nearest metropolitan hub** is named.
- The panel streams that point's **33-D local-sky matrix** (all 11 bodies, North/East/Up) and
  its **L\*a\*b\* + HEX** colour signature — computed by the reused v6 ephemeris + SIREN, so
  the swatch matches the field on the globe.

### Keyboard
`Space` play/pause · `L` now · `←/→` step · `↑/↓` zoom · `G` graticule · `R` reset view.

---

## 4 · How it fits together

The globe fragment shader is a single `texture2D` lookup (plus `asin`/`atan` for the uv) —
there is **no SIREN and no ephemeris in the render path**, which is what removes the v6 lag.
All the physics happens once per Julian Date in the worker:

1. `equatorialDirs(jd)` / `gmstRad(jd)` (reused from v6) give the 11 bodies' geocentric
   equatorial directions — computed once per bake, not per node.
2. Each grid node does the cheap horizontal projection + the SIREN forward + the bounded
   L\*a\*b\* head + L\*a\*b\*→sRGB (all reused from v6).
3. The RGBA grid is transferred to the main thread and uploaded as an equirectangular
   `DataTexture` with **bilinear filtering** — the interpolation between nodes is the
   "macro-scale regional" smoothing.

Because the worker owns the heavy loop, the main thread only ever renders a texture: 60 fps,
independent of zoom and window size. City queries and the HUD evaluate the SIREN *exactly* at
the point in question, so accuracy is never interpolated where it matters.

---

## 5 · Verifying

```bash
python -m pytest tests/test_version7.py -q      # cities, sampler, train->export bundle
```

The frontend field baker reuses the v6 modules, which are parity-checked against the Python
engine (float32-exact ephemeris, ~1e-6 SIREN); the equirectangular node colours reproduce the
Python reference exactly (0/255 per channel).

---

## 6 · Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| Blank page / `import` errors for `../../version6/…` | You served `version7/web` directly. Serve the **repo root** so the v6 imports resolve (§3). |
| "no weights.json — showing an UNTRAINED field" notice | Train + export first (§1–§2), then reload. |
| Field looks flat / grey | Under-trained — the isometric loss grows chroma slowly. Train tens of thousands of steps and re-export, or raise `--color-scale`. |
| `field N/s` is low on a huge grid | Pick a coarser **Field detail**; the globe still renders at 60 fps, only the field-refresh rate changes. |
| Blank page / WebGL error | Browser lacks WebGL2 — use a current desktop browser. |

---

## 7 · File map

```
version7/
  cities.py           curated metropolitan hubs (name, lat, lon, region)
  config.py           V7Config: reuses v6 SirenConfig + grid/data/train knobs
  dataset.py          StructuredNodes: cities + regional lattice + uniform, across the timeline
  training.py         train loop (reuses v6 SIREN/loss/schedule) + export helpers
  train.py            CLI: python -m version7.train
  export.py           CLI: python -m version7.export  (weights + cities + manifest)
  web/
    index.html        page shell (Temporal Helm + Observer HUD)
    style.css         glass-panel styling
    field.js          bake the regional field grid (reuses v6 ephemeris6 + siren6)
    fieldworker.js    background worker that runs field.js off the main thread
    globe.js          Three.js texture-mapped globe + city markers + picking
    timecal.js        Julian Date <-> calendar + display-timezone helpers
    main.js           app: Temporal Helm, timezone, zoom, opacity, HUD, keyboard
    weights.json      (generated) SIREN weights
    cities.json       (generated) hub markers
    manifest.json     (generated) grid + timeline + architecture
```

Train → export → serve the repo root → open `version7/web/`.
