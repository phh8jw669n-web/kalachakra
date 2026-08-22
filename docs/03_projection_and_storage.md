# 03 — Analytical projection and binary storage

## Projection `G(t) → E(t, s)`

`kalachakra.projection.spatial.project` turns one global frame into the local
field over every observer with closed-form spherical trigonometry — no ephemeris
call per node. The steps, all vectorized over `(nodes × bodies)`:

1. **Decode** ecliptic `(λ, β)` from the unit-vector columns of `G(t)`.
2. **Obliquity** `ε(jd)` (IAU 1980 mean obliquity) → **equatorial** `(α, δ)`
   via the standard rotation.
3. **Local sidereal time** per node: `LST = GMST(jd) + longitude_east`; the RA of
   the meridian `RAMC = LST`.
4. **Hour angle** `H = LST − α`, then

   ```
   sin h = sin φ sin δ + cos φ cos δ cos H            (altitude)
   θ = atan2(−cos δ sin H,  sin δ cos φ − cos δ sin φ cos H)   (azimuth, from N, +E)
   ```
5. **Ascendant** longitude from `RAMC`, `ε`, `φ`:

   ```
   λ_asc = atan2( cos RAMC,  −(sin RAMC cos ε + tan φ sin ε) )
   ```
   and the **offset** `Δφ = wrap(λ_body − λ_asc)`.
6. **Encode** `e_i(s,t) = [cos θ cos h, sin θ cos h, sin h, cos Δφ, sin Δφ]`.

The numpy implementation is the correctness oracle for the on-device Metal
kernel; both compute the same broadcast. Validated by `tests/test_projection.py`
(unit sub-vectors, altitude range, deterministic output, decode round-trip).

> Edge case: near the polar circles the Ascendant is ill-conditioned; the
> `(cos, sin)` encoding keeps the field finite and continuous there.

## Binary storage (`kalachakra.storage`)

The global-state timeline is serialized to contiguous `.mmap` chunks:

- **BF16** — `float32_to_bf16` truncates to the top 16 bits with
  round-to-nearest-even (numpy has no native bf16). Exact for
  powers-of-two-ish values; ~2–3 significant decimal digits otherwise.
- **Delta encoding** — `delta_encode` stores frame 0 absolutely and successive
  frame-to-frame differences; the smooth planetary trajectories make these tiny,
  which is what takes the raw ~1.9 TB down to ~300 GB.
- **`EphemerisStore`** — chunk files + `manifest.json`; `write_chunk` /
  `read_chunk` round-trip through `numpy.memmap`.
- **`RingBuffer`** — a background thread prefetches upcoming chunks with bounded
  depth (back-pressure = automatic eviction), so the GPU never waits on disk.

Round-trips are validated by `tests/test_storage.py`.
