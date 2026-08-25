# 00 · Conceptual Guide

*The intuition first. No code here — just the philosophy, the shapes of the data,
and what a person actually sees.*

---

## 1. The core idea

Conventional astrology software is a **lookup table**: it reduces the sky to a
handful of named angles ("Mars square Saturn") and prints canned text. Kalachakra
throws that away. It treats the solar system as a set of **continuous wave
generators** — every planet is a moving source, and what matters is the *geometry
of their interference* at each instant, everywhere on Earth.

From that geometry the system computes **objective mathematical quantities** only:
angular separations, harmonic resonance, structural tension, circular
concentration, temporal derivatives, latent norms. It never emits an
interpretation. The meaning, if any, is the reader's; the machine's job is to
measure the field and render it faithfully.

Three commitments make this work and recur everywhere in the code:

- **Native units, no human grid.** Time is sampled in **Vighatikas** (24 seconds),
  a step at which Earth's eastern horizon advances ≈0.1° per frame — so ingestion
  is locked to the planet's real rotation, not to the 60‑minute hour. Space is a
  geodesic mesh where "distance" is angular separation, not a 360° lat/lon grid.
- **Boundary‑free encoding.** Every cyclic angle enters the math as `(cos, sin)`
  or a unit vector. There is never a discontinuity at 0°/360°. A planet at 359.9°
  and one at 0.1° are neighbors, as they should be.
- **Global/local decoupling.** The planets' positions are computed **once** per
  instant (the *global state*). Turning that into "what the sky looks like from a
  specific point on Earth" is a pure geometric projection — no per‑location physics
  loop. This single decoupling is what lets the system scale to the whole globe.

---

## 2. Celestial geometry vs. terrestrial space

The single most important concept in the whole project is the split between **the
sky** and **the ground**.

**The sky is global.** At a given instant, each body has an ecliptic longitude and
latitude that are *the same no matter where you stand on Earth* (planets are
effectively at infinity; the Moon has a tiny parallax). So the entire
configuration of the solar system at time `t` is a small, location‑free object —
in the original pipeline a `10 × 7` matrix `G(t)`; in the decoupled engine a
`10 × 5` celestial tensor compressed to a single 512‑D **tension vector**.

**The ground is local.** What *changes* from place to place is the **Ascendant and
the houses** — which part of the sky is rising on your local horizon. That depends
on your latitude and the local sidereal time (your longitude + Earth's rotation).
This is the only thing that makes London's field differ from Tokyo's at the same
instant.

Both architectures are built around this split:

```
        the sky (global, one per instant)              the ground (local, per point)
   ┌───────────────────────────────────────┐     ┌────────────────────────────────────┐
   │  planetary positions & velocities      │     │  observer latitude / longitude     │
   │  G(t): 10 bodies × 7 features          │ ──► │  local horizon geometry / Ascendant│
   │  (or 10 × 5 celestial tensor)          │     │  → the field value at that point    │
   └───────────────────────────────────────┘     └────────────────────────────────────┘
                    compute once                          broadcast / query many
```

- In the **discrete pipeline**, "broadcast" means projecting `G(t)` onto 122,880
  fixed mesh nodes at once (`E(t,s)`), then compressing with a neural net.
- In the **decoupled engine**, "query" means: run the Sky Encoder once to get the
  tension vector, then evaluate the Earth Lens at *any* `(lat, lon)` you like — one
  point, a city, or a full‑globe grid — on demand.

Why does this matter? Because it separates **what the solar system is doing**
(expensive, computed once) from **where you are looking** (cheap, evaluated
anywhere). The sky is shared; the lens is personal.

---

## 3. Macro and micro timelines

The timeline is a **closed 10,256‑year manifold**, anchored at the Kali‑Yuga epoch
(3102 BCE, Julian Day 588465.5) and running to 7154 CE. Users experience it at two
very different scales, and the system is explicitly designed for both:

- **Macro (millennia).** Slow bodies — Jupiter, Saturn, the lunar nodes, precession
  — drift over centuries. Zoomed out, the energy field breathes slowly: generational
  patterns, great conjunctions, the long precessional wobble. Sampling here can be
  coarse (a day or a year per step) because nothing fast is happening.

- **Micro (minutes).** The Moon moves ~0.5°/hour and the Ascendant sweeps the whole
  zodiac every day, so at fine scale the field *shimmers* — rapid lunar transits and
  Ascendant crossings produce sharp, local structure. Resolving these needs the
  native 24‑second step.

The project handles this multi‑scale nature in several concrete ways:

- **Adaptive time‑stepping** (Great Indexer): the clock cruises at 1‑hour steps and
  automatically downshifts to 24‑second micro‑frames whenever the geometry starts
  moving fast, then stabilizes back.
- **Curriculum learning** (base‑model training): training starts coarse (24‑hour
  stride, whole‑timeline sweeps) and progressively refines to 24‑second micro‑bursts,
  so the model learns the slow background before the fast shears — without wasting
  compute re‑learning a near‑static sky billions of times.
- **Temporal mipmaps** (serving): the index stores three tiers — native, hourly,
  daily‑epochal — and a query router picks the right resolution for the requested
  time span, so a millennium‑wide viewport doesn't page in billions of frames.

---

## 4. What a user actually experiences

Depending on the entry point, a user meets the system as one of several concrete
things:

- **A cosmic‑weather reading** (`kalachakra reading`). Point at a place and time and
  get the real planetary positions plus objective signatures — harmonic resonance,
  structural tension, geometric potential, eclipse proximity, the dominant aspects —
  computed directly from the ephemeris, *no model required*. It correctly finds real
  eclipses (e.g. the 2024‑04‑08 total eclipse: Sun–Moon 0.04°).

- **A live global energy field** (`web/decoupled.html`). The decoupled model's
  learned signature is painted as a **colored layer over a world map** and animated
  through time. You watch the field evolve across the whole globe, scrub the
  timeline, and click any point to inspect its exact color and a **per‑planet
  attention attribution** (which bodies drive the tension there).

- **A kinetic radar** (`web/radar.html`). The isomorphic transducer renders the
  latent field as a physically‑grounded optical state — radiant flux, Planckian
  color temperature, an orthonormal temporal spectrum, and a fluid vector field —
  on a globe plus a regional micro‑canvas. Uniquely, this rendering is **losslessly
  invertible**: the exact 64‑D latent can be recovered from the picture.

- **An archetype dossier** (the Great Indexer → `dossiers.sqlite`). Each of the 4096
  discrete archetypes gets an 18‑profile "personality file" — its magnitude,
  isolation, spatial drift, orbital attribution, temporal periodicity, ecosystem
  relations — queryable in milliseconds with no PyTorch.

- **A twin search** (`web/kundali.html`). Given a birth chart, the Kundali engine
  sweeps history for days whose sidereal geometry matches at eight escalating tiers,
  and — for house‑dependent tiers — solves the `(lat, lon)` curve on Earth where the
  match would occur.

---

## 5. End‑to‑end data flow (both pipelines)

```
                         ┌──────────────────────────────────────────┐
   Swiss Ephemeris ─────►│  G(t)  global state  (10 bodies × 7)      │  once per instant
   (Moshier/Swiss/JPL)   └──────────────────────────────────────────┘
                                     │                     │
             ┌───────────────────────┘                     └────────────────────────┐
             ▼  DISCRETE VQ-MESH PIPELINE                          ▼  DECOUPLED ENGINE
   ┌───────────────────────────┐                        ┌───────────────────────────────┐
   │ project → E(t,s)          │  122,880 nodes         │ celestial tensor (10 × 5)     │
   │ (topocentric, parallax)   │                        │ → Sky Encoder (transformer)   │
   └───────────────────────────┘                        │ → 512-D global tension vector │
             ▼                                           └───────────────────────────────┘
   ┌───────────────────────────┐                                     │
   │ Spherical AE + STFNO       │  64-D latent z(t,s)                 ▼
   │ → VQ codebook (4096)       │  discrete tokens        ┌───────────────────────────────┐
   └───────────────────────────┘                          │ Earth Lens (implicit field)   │
             ▼                                             │ (tension, lat, lon) → OKLab   │
   ┌───────────────────────────┐                          └───────────────────────────────┘
   │ signatures / rarity /      │                                     │
   │ Great Indexer / transducer │                                     ▼
   │ → Parquet / DuckDB / SQLite│                          ┌───────────────────────────────┐
   └───────────────────────────┘                          │ live global texture + pinpoint│
             ▼                                             │ query + attention attribution │
   ┌───────────────────────────┐                          └───────────────────────────────┘
   │ REST / gRPC / WebSocket    │
   │ → WebGL globe / radar      │
   └───────────────────────────┘
```

The rest of `docs2/` fills in every box precisely: the exact tensors, the math of
each loss, every hyperparameter, and every endpoint. Continue with
[`01-architecture-overview.md`](01-architecture-overview.md).
