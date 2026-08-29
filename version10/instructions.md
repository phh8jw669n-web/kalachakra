# Kalachakra Version 10 — The Astrological Anchor (13-Token Engine)

Version 10 extends the v9 Topocentric Self-Attention engine from an 11-token *astronomical*
observer to a 13-token *astrological* one by adding two fast-moving structural anchors — the
**Ascendant (ASC)** and **Midheaven (MC)**. These are observer-specific points that sweep the
whole sky in a day, so they anchor the slow planetary bodies to a high-speed geographic grid:
the attention matrix (`QKᵀ`) can organically resolve sharp astrocartography-style lines without
mathematical ringing. Fully self-contained (no cross-version imports).

```
(lat, lon, jd) ─► 13 tokens ×[N,E,Zenith] ─► self-attention ─► pooled read-out ─► OKLCH (C,H) ─► sRGB
   any float     11 bodies + ASC + MC        Qkᵀ relations     per-token energy   PURE chroma    to a field texture
```

| Module | What it is | Where it lives |
| --- | --- | --- |
| **1 · Backend** | 13-token state (+ ASC/MC astronomy), softer-gate + TV loss, model, export | `ephemeris.py`, `state.py`, `attention.py`, `losses.py`, `dataset.py`, `training.py`, `train.py`, `export.py` |
| **2 · GPU render** | the whole 13-token net once per time step into a **field texture** (ASC/MC computed per texel), sampled by the globe | `web/shader10.js`, `web/main.js` |
| **3 · UI / Helm** | LIVE/Time-Machine clock, orbiting bodies, **ASC/MC markers**, world map, controls, energy HUD | `web/main.js`, `web/planets10.js`, `web/ephemeris10.js`, ... |

---

## 1 · The two anchors (what v10 adds)

* **Token 11 — Ascendant (ASC):** the ecliptic point rising on the observer's **eastern
  horizon**. As a topocentric vector it sits on the horizon (Zenith ≈ 0), toward the East.
* **Token 12 — Midheaven (MC):** the ecliptic point crossing the observer's **local meridian**
  (upper culmination). As a topocentric vector it lies in the meridian plane (East ≈ 0), above
  the horizon.

Both are computed from standard spherical trig and **verified against `pyswisseph` `swe.houses`
to < 0.01°** (`ephemeris.py::asc_mc_ecliptic`, mirrored in `ephemeris10.js`):

```
RAMC   = LST = GMST + lon
λ_MC   = atan2( sin RAMC , cos RAMC · cos ε )
λ_ASC  = atan2( cos RAMC , −(sin RAMC · cos ε + tan φ · sin ε) )
```

Crucially, unlike the 11 geocentric bodies (whose directions are observer-independent), **ASC
and MC depend on the observer's lat/lon** and sweep the sky in ~24 h — so they inject strong,
sharp *spatial* structure into the field. Latitude is clamped to ±89.99° (finite `tan φ`, pole
stability) in Python, JS and the shader alike.

State dimensions grow accordingly: **39-D local** (13×3) and **78 gated chords** (`C(13,2)`).

---

## 2 · Why attention, and the loss (unchanged core, recalibrated for v10)

The architecture is v9's: each token is a `[North, East, Zenith]` vector plus a learned
body-identity embedding; a single-head block computes `softmax(Q·Kᵀ/√d + vis_bias·zenith)·V`
(observer-dependent because the learned bilinear form is not rotation-invariant, and horizon-
visibility-primed so above-horizon tokens dominate). The isometric loss targets a **fixed,
observer-dependent** geometric distance (local vectors + horizon-gated chords) — never the
network's own attention (that collapses). Colour is pure **OKLCH** (see §3).

**Structural anchors are exempt from the horizon prior (`n_anchors = 2`).** The `vis_bias·zenith`
term is applied only to the 11 *body* tokens; the last two tokens (ASC, MC) are coordinate axes,
not physical bodies, so they get the full `vis_bias` regardless of altitude (their zenith is
treated as 1). This is mandatory: the Ascendant sits *on* the horizon by definition (zenith ≈ 0),
so gating it by zenith would wrongly zero out its prominence — after the exemption ASC competes
on equal footing with MC in both the per-block attention and the pooled read-out. The exemption
is mirrored identically in `attn10.js` and the GLSL field shader (parity-tested).

**Anti-ringing recalibration (v10):** softer horizon gate — `gate_k` 8.0 → **3.0**. A gentle
falloff for angular proximity avoids a near-discontinuous target; v10's sharpness comes from the
fast ASC/MC tokens instead of a steep gate.

**v10.1 — the beaded-zipper cure (hue winding).** Under *deep* training (50k+ steps) a dotted
rainbow "zipper" can appear along an astrocartography line. It is **not** real structure and it is
**not** softmax saturation: the diagnostic shows the hue angle *winding* through the colour wheel
several times across a few degrees, driven by the head weights inflating. Because the isometric
objective depends only on the *local* colour gap `‖Δ(a,b)‖`, it is blind to adding whole 2π turns
to the hue — so the network is free to wind at zero loss cost, yet winding encodes **no signal**.
The cure removes only that gauge freedom, never the signature:

* **Bounded cosine attention** (`qk_norm`, default on) — Q, K (and the pool query/keys) are
  L2-normalised, so the pre-softmax logit is a cosine in `[-1,1]` times a learnable temperature
  clamped to ≤ `attn_temp_max` (30). Deep training can no longer inflate Q·K into a saturated
  switch. Sharp *true* edges (a high temperature) stay reachable; only the runaway is outlawed.
* **Moderate weight decay** — `3e-3` (was `1e-4`) on the 2-D weight matrices only (never biases,
  temperatures, `q_pool`, or `E_body`). Holds the head in the small-number regime that renders
  solid lines. Deliberately far below the PRD's `0.1`, which would dull real signal.
* **Isometry-referenced anti-winding term** (replaces the blunt TV smoothness) — the *same*
  isometric objective enforced at fine + coarse spatial scale: neighbouring observers' colours
  must differ by exactly `γ·d_sky(pair)`, no more. Winding grossly overshoots that (a full hue
  turn while `d_sky` barely moves) and is removed; a genuine gradient already satisfies it and is
  untouched. This is faithful *by construction* — the reference is the true sky metric, not zero.

Rejected from the source PRD after checking: **softmax-saturation** as the diagnosis (false — the
40k model's logits max ≈ 2, softmax max-prob ≈ 0.4), **weight_decay 0.1** (over-smooths), and
**Huber loss** (irrelevant to hue winding).

```
d_local = ||ΔV||/√39 ; d_rel = ||ΔR||/√78 (R = horizon-gated chords, k=3)
d_sky   = 0.5·d_local + 0.5·d_rel
L = MSE( ||Δ(OKLab a,b)|| , γ·d_sky ) + λ_anchor·anchor
      + Σ_scale λ_scale·MSE( ||Δ(a,b)||_neighbour , γ·d_sky_neighbour )     (γ = 0.35)
```

---

## 3 · Colour: OKLCH (perceptually uniform, bead-free)

The head outputs polar OKLCH — `C = 0.4·sigmoid(z0)`, `H = z1` (raw radians, cyclic via
cos/sin) — carried as OKLab `(a,b) = (C·cosH, C·sinH)`. Euclidean distance on `(a,b)` *is* the
OKLCH cylindrical distance, so the loss is perceptually uniform. The **field texture stores
`(a,b)`** (half-float where available) and the globe reconstructs **OKLab → sRGB per pixel**
with a hue- and lightness-preserving gamut clip (bisection to the sRGB boundary). Interpolating
the Cartesian `(a,b)` — never a hue angle — is what keeps gradients smooth and bead-free.

---

## 4 · Train

```bash
python -m version10.train --steps 40000 --batch 2048 --export version10/web/weights.json
```

| Flag | Default | Effect |
| --- | --- | --- |
| `--d-model` / `--d-ff` / `--d-head` / `--blocks` | 32 / 64 / 32 / 2 | attention size. |
| `--gamma` | 0.35 | chroma scale (OKLab units). |
| `--w-local` / `--w-rel` | 0.5 / 0.5 | local vs gated-chord weight. |
| `--gate-k` | **3.0** | horizon-gate steepness (softer than v9's 8.0). |
| `--tv-weight` / `--tv-delta` | **0.05** / 0.75° | spatial-smoothness weight / neighbour offset. |

`export` writes `weights.json` (attention weights + head/target metadata). Inference is a plain
forward, so the loss-only scales are provenance.

---

## 5 · Run

```bash
cd version10/web && python -m http.server 8080     # open http://localhost:8080/ (WebGL2)
```

### GPU model
The CPU uploads the **11** Earth-fixed body directions, plus **GMST and cos/sin ε**, once per
time step. The **field pass** (offscreen equirectangular texture, once per time change) then per
texel: builds the horizon basis (latitude clamped off the pole), copies the 11 bodies and
**computes the ASC & MC per texel** from GMST/ε/lat/lon (identical maths to Python, verified on
the real GPU), runs the whole 13-token net, and writes the OKLab `(a,b)`. The **globe pass**
samples that texture and converts to sRGB per pixel — so rotating/zooming is a cheap texture
lookup that can never overload the GPU. The **Render** selector sets field resolution
(192×96 / 320×160 / 512×256).

### Controls & UI
All of v9's helm (LIVE / Time-Machine, Play, Now, clock zone, speed, step presets + custom step,
scrubber, Jump-to, zoom, field opacity, render quality, overlays, keyboard shortcuts), plus:
- **ASC/MC markers** — a gold **crosshair** for the MC and a cyan **vertical bar** for the ASC,
  placed over each anchor's sub-point for the pinned observer (toggle: **ASC/MC** overlay). Move
  the pin or the time and they move — so you can see which region a sharp chromatic line anchors.
- **Observer HUD** — click the globe to pin a point; it streams the 13-token energy contribution
  (attention pool weights, incl. ASC/MC), the 13-token local matrix, and the OKLCH C/H°/HEX.

---

## 6 · Verifying

```bash
python -m pytest tests/test_version10.py -q
```

Checks the 13-token geometry (ASC on the horizon, MC on the meridian), **ASC/MC vs pyswisseph
houses (< 0.05°)**, their observer-dependence, the 78 gated chords, the OKLCH head, the TV term,
and the export→re-run parity. Verified beyond the unit tests: the JS ephemeris reproduces Python
to **4e-16**, and the **GLSL field shader — including the per-texel ASC/MC — matches the JS/
PyTorch pipeline on a real headless WebGL2 GPU** to 8-bit precision (weights are preserved; no
retraining needed for render changes).

---

## 7 · File map

```
version10/
  ephemeris.py     analytic ephemeris + ASC/MC (asc_mc_ecliptic), 13 tokens
  state.py         39-D local + 78 horizon-gated chords (loss target)
  attention.py     single-head self-attention, 13 tokens, OKLCH head
  losses.py        isometric loss + tv_loss (spatial smoothness)
  config.py        AttnConfig(n_bodies=13) / DataConfig / TrainConfig(gate_k=3, tv_weight)
  dataset.py / training.py / train.py / export.py
  web/
    index.html style.css main.js
    shader10.js    field (13 tokens, per-texel ASC/MC) + globe (OKLab->sRGB) shaders
    attn10.js      JS attention forward + OKLCH head + OKLab->sRGB
    state10.js     JS 39-D local + 78 gated chords
    ephemeris10.js JS ephemeris incl. ASC/MC (parity with ephemeris.py to 4e-16)
    planets10.js   11 bodies as spheres + ASC/MC anchor markers (crosshair / bar)
    geo.js timecal.js coastlines.json
    weights.json golden.json   (generated)
```

Train → export → serve `version10/web/` → explore.
