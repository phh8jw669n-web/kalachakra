# 12 · Glossary

Terminology used across the project, grouped by area. Cross‑references point to the
document with the full treatment.

---

## Timeline & units

- **Vighatika** — the native temporal step, 24 SI seconds. At this rate Earth's
  horizon advances ≈0.1°/frame. (`VIGHATIKA_SECONDS`.) → [02](02-ephemeris-and-geometry.md)
- **Kali Yuga epoch** — the timeline anchor, 3102‑02‑18 BCE 00:00 UTC = **JD 588465.5**
  (frame 0).
- **Timeline** — a closed 10,256‑year manifold, 3102 BCE → 7154 CE, ≈1.349×10¹⁰
  frames.
- **Julian Day (JD)** — continuous day count. Convention here: integer JD = **noon
  UT**, `.5` = midnight.
- **Astronomical year numbering** — year 0 = 1 BCE (used by the calendar module).
- **Macro / micro timeline** — millennia (slow bodies) vs. minutes (Moon/Ascendant);
  the system samples both via adaptive stepping / curriculum / mipmaps. → [00 §3](00-conceptual-guide.md)

## Ephemeris & backends

- **`G(t)`** — the global state vector, `10 bodies × 7 features`
  `[cosλcosβ, sinλcosβ, sinβ, λ̇, β̇, r, ṙ]`. Location‑free. → [02 §4](02-ephemeris-and-geometry.md)
- **`E(t,s)`** — the local projected field, `N nodes × 10 × 5`
  `[cosθcosh, sinθcosh, sinh, cosΔφ, sinΔφ]`. → [03 §2](03-projection-grid-storage.md)
- **Moshier / Swiss / JPL** — the three `pyswisseph` backends. Moshier is file‑free
  (~3000 BCE–3000 CE); Swiss `.se1` and JPL **DE441**/DE431 cover the full span.
- **Ephemeris speed flag** — swe flag that also returns velocities (`λ̇`).

## Celestial geometry

- **Ecliptic longitude/latitude (λ, β)** — a body's position on the ecliptic.
- **Ayanamsha** — the precession offset between the tropical and sidereal zodiacs;
  the project uses the **Lahiri** ayanamsha for sidereal work; it is also a row of
  `G(t)`. → [10](10-kundali-engine.md)
- **Sidereal vs. tropical** — sidereal = tropical − ayanamsha (fixed‑star frame).
- **RAMC** — Right Ascension of the Meridian = GMST + geographic longitude.
- **Ascendant / Lagna** — the rising ecliptic point on the local horizon; depends on
  latitude and local sidereal time (the only location‑dependent quantity).
- **Nakshatra** — one of 27 lunar mansions (arc 13°20′).
- **Navamsa (D9)** — a ninth‑harmonic divisional sign; element rule `(sign·9+idx)%12`.
- **Whole‑sign house** — `(planet_sign − asc_sign) % 12 + 1`.
- **Aspect / orb** — angular relationship (conjunction 0°, sextile 60°, square 90°,
  trine 120°, opposition 180°) within an allowed orb.
- **Stellium** — a concentration of bodies in one sign (circular concentration).
- **Topocentric / parallax** — corrected for the observer's position on Earth's
  surface (vs. geocentric); resolves lunar parallax so eclipses localize.

## Spatial

- **Geodesic mesh** — the 122,880‑node spherical observer grid; distance = angular
  separation. → [03 §1](03-projection-grid-storage.md)
- **Fibonacci sphere** — the near‑uniform lattice used to realize exactly 122,880
  nodes (an icosphere gives `10·4ⁿ+2`, never 122,880).
- **H3** — Uber's hexagonal geospatial index (spatial predicate for the DuckDB
  router). → [03 §4](03-projection-grid-storage.md)
- **Micro‑grid** — an on‑the‑fly regional lat/lon patch for continuous LOD.

## Neural / ML

- **STFNO** — Spatio‑Temporal Fourier Neural Operator: geodesic conv (space) + 1‑D
  spectral conv (time). → [05](05-models.md)
- **FNO / spectral conv** — learns an operator in the Fourier domain (rfft → complex
  weight on the lowest `modes` → irfft); discretization‑invariant.
- **Geodesic convolution** — isotropic message passing over `k` great‑circle
  neighbors (`self_lin(x) + neigh_lin(mean(neighbors))`).
- **Latent `z(t,s)`** — the continuous 64‑D bottleneck (`LATENT_DIM`).
- **VQ / codebook / perplexity** — Vector Quantization maps the latent to the nearest
  of `K` codebook vectors (v3: 4096, cosine, EMA); **perplexity** = `exp(entropy of
  the assignment histogram)` (codebook utilization).
- **RVQ** — Hierarchical Residual VQ: macro code then macro‑conditioned micro code;
  `leaf = macro·64 + micro` (4096 archetypes).
- **Archetype / token** — one discrete geometric state (a codebook leaf).
- **Straight‑through estimator** — passes gradients through the (non‑differentiable)
  quantization by copying them from `z_q` to `z_e`.
- **INR** — Implicit Neural Representation: a coordinate‑input MLP producing a
  continuous field (the Earth Lens). → [09 §6](09-decoupled-engine.md)
- **Random Fourier Features (RFF)** — `[sin(2π x B), cos(2π x B)]`; defeats MLP
  spectral bias so high‑frequency detail resolves.
- **SIREN / sine activation** — periodic activations for INRs (with special init).
- **Gaussian activation** — `exp(−x²/2σ²)`, the Earth Lens default.
- **Tension vector** — the Sky Encoder's 512‑D L2‑normalized global summary.
- **Planetary attribution** — the CLS token's attention onto each body (which planets
  drive the current tension).
- **OKLab** — a perceptually‑uniform color space `(L, a, b)`; the Earth Lens output.
- **Lion** — a memory‑light optimizer (sign‑of‑momentum, decoupled weight decay).
- **Cosine annealing with warm restarts** — LR schedule that periodically resets and
  re‑anneals (`T_0`, `T_mult`).
- **AMP / BF16 / FP16** — mixed precision; bf16 on CUDA, fp16 on MPS.
- **Curriculum / micro‑bursting** — progressive temporal‑stride training; sub‑hour
  strides sample random calendar windows instead of a full sweep. → [04 §2](04-data-pipelines.md)

## Signatures & analysis

- **Geometric potential (`‖z‖`)** — mass‑convergence intensity.
- **Temporal shear (`‖dz/dt‖`)** — rate of phase transition.
- **Harmonic resonance** — weighted aspect‑kernel field (constructive interference).
- **Structural tension** — frame‑to‑frame change of the geometry.
- **Singularity** — a robust‑threshold outlier of maximum tension. → [07 §4](07-inference-and-analysis.md)
- **Rarity index** — normalized‑NLL statistical uniqueness of a token over deep time.
- **HDBSCAN** — density clustering of the latent manifold.
- **Resonance Divergence Metric** — the testing daemon's benchmark score.

## The Great Indexer

- **Dossier (`dossiers.sqlite`)** — the per‑archetype profile database (18 profiles ×
  4096 tokens). → [07 §8](07-inference-and-analysis.md)
- **Adaptive clock** — cruises at 1 h, downshifts to 24 s on fast geometry.
- **Domains 1–5** — tensor physics / orbital / spatial / temporal / ecosystem.
- **`--lite`** — skip Domain 5 + Domain‑1 PCA for rapid UI prototyping (NULL columns).

## Transducer

- **Isomorphic transducer** — a losslessly invertible physics‑based renderer.
  `invert(transduce(...))` recovers the latent to machine precision. → [07 §7](07-inference-and-analysis.md)
- **Naka‑Rushton** — a saturating (here inverted to boundless) magnitude→flux curve.
- **Planckian locus / CCT** — black‑body color as a function of temperature (encodes
  rarity).
- **Helmholtz‑Hodge / LIC** — decompose a vector field into curl‑ and divergence‑free
  parts; Line Integral Convolution visualizes flow.
- **Spherical harmonics (SH)** — the orthonormal basis mapping the latent to Earth
  topography (exact Gauss‑Legendre quadrature).

## Storage & serving

- **BF16 + delta encoding** — the compact `.mmap` timeline format.
- **Ring buffer** — async chunk prefetcher overlapping decode with compute.
- **Temporal mipmap** — native / hourly / daily‑epochal tiers for wide time spans.
- **Parquet / DuckDB** — century‑partitioned columnar store + in‑process query router.
- **Broadcast engine** — per‑coordinate metric server over the mesh.
- **`PROJECTION_VERSION`** — the stamped semantics of `E(t,s)` (2 = topocentric);
  incompatible artifacts refuse to mix.
