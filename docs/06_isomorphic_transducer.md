# 06 — The Isomorphic Mathematical Transducer

The rendering layer is an **objective sensory converter**, not a data
visualization: mathematical type binds to physical optical type, and the optical
state is a **losslessly invertible** storage medium for the latent field. All of
this is implemented and unit-tested in `kalachakra.transducer`; the WebGL client
(`web/radar.html`) mirrors the same math.

## The isomorphic law (Page 1)

| Mathematical object | Physical channel | Module |
|---|---|---|
| scalar magnitude `‖z‖` | radiant flux (boundless) | `photometric.naka_rushton` |
| scalar rarity (NLL) | colour temperature (Planckian) | `photometric.rarity_to_temperature`, `planckian_xy` |
| 4 temporal band energies | visible spectrum (380–750 nm) | `spectral.SpectralTransducer` |
| vector field (div / curl) | fluid kinematics | `kinematics.helmholtz_hodge`, `field_from_sources`, `line_integral_convolution` |
| 64-d latent tensor | topography | `topography` (spherical harmonics) |

## Lossless invertibility (the validation)

`state.IsomorphicTransducer.transduce(...)` → `OpticalState`; `invert(...)`
reconstructs the inputs. The test `test_full_transducer_recovers_latent_to_machine_precision`
proves the round trip:

- **latent** recovered to `< 1e-8` (orthonormal SH; raw round trip `~4e-15`),
- **band energies** to `< 1e-9` (orthonormal spectral bases),
- **potential** to `< 1e-6` (Naka-Rushton bijection),
- **rarity** to `< 1e-9` (temperature is an exact monotone function of rarity),
- **vector div/curl** to `< 1e-9` (Helmholtz-Hodge on Nyquist-free grids).

Nothing is destroyed by clipping (Naka-Rushton is boundless, additive blending),
band interference (orthogonal bases), or geometry loss (orthonormal harmonics).

## Scalar → light (Page 2)

- **Flux**: `flux = x/(x+k)` maps `[0,∞)→[0,1)` with exact inverse — a 10 %
  change in tension is a proportional change in photon flux whether the baseline
  is a void or a stellium.
- **Temperature**: rarity maps log-uniformly to 1200 K (common, red/IR) … 40000 K
  (singular, UV) along the Planckian locus; `cct_from_xy` (McCamy) is the
  pixel-space inverse the shader uses within its valid range.
- **Spectrum**: four orthonormal bases over 380–750 nm; the rendered chromaticity
  is their additive superposition, and the four scalar energies stay separable.

## Vector → fluid (Page 3)

Helmholtz-Hodge splits the field into irrotational (divergence = applying vs.
separating; sinks converge, sources radiate) and solenoidal (curl = orthogonal
shear = vorticity). `field_from_sources` builds a field from div/curl and
`divergence`/`curl` recover them. `line_integral_convolution` renders the flow
densely (every pixel encodes streamline orientation) instead of sparse particles;
advection speed is bound to the temporal derivative (stations → stagnation).

## Tensor → topography (Page 4)

The 64 latent dims are exactly degrees l=0..7 of the real spherical harmonics
((7+1)²=64). The Earth mesh height is `Σ cᵢ Yᵢ(θ,φ)`; because the basis is
orthonormal, quadrature analysis inverts the extrusion back to the exact
coefficients. Crests, trenches, and nodal zero-crossings emerge only from the
wave equations — no templates.

## Gestalt + Sidebar Inspector (Page 5)

The client renders the full field as one perceptual gestalt, while a double-click
opens the Sidebar Inspector, which calls `/telemetry` and deconstructs the state
into raw numerics: timestamp, rarity percentile, VQ archetype, the four band
energies, each body's Cartesian unit vector + radial distance + angular velocity,
and the applying/separating flag. Shift+double-click opens the regional
micro-canvas (`/microgrid`, §4 continuous LOD).

## Status

The transducer math is complete and tested to machine precision. `web/radar.html`
implements the same mappings (SH topography, blackbody + Naka-Rushton + spectral
optics, Helmholtz/LIC fluid, diurnal sweep, dual viewport, inspector); as with any
WebGL, it is not unit-tested in this environment — the Python transducer is its
oracle. Requires scipy for the topography channel (`pip install scipy`).
