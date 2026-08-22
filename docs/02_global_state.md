# 02 — Generating the global state `G(t)`

`G(t)` is produced by `kalachakra.ephemeris.global_state` from the Swiss
Ephemeris (a wrapper over NASA JPL **DE441**). This note covers the encoding and
the one real operational hurdle: installing the ephemeris data.

## The 7-D per-body encoding

For each entity the raw ecliptic state `(λ, β, r)` and its rates
`(λ̇, β̇, ṙ)` are packed into a smooth, boundary-free vector:

```
v_i(t) = [ cos λ cos β,   # \
           sin λ cos β,   #  |- 3D unit direction (no 0/360° seam)
           sin β,         # /
           λ̇, β̇,          # angular velocities (rad/day)
           r,             # radial distance (AU)
           ṙ ]            # radial velocity (AU/day)
```

Stacking the ten rows (Sun … Saturn, Rahu, Ketu, Ayanamsha) gives `G(t)`.

- **Rahu / Ketu** come from the true lunar node; Ketu is the node + 180°
  (`bodies.CelestialEntity.longitude_offset_deg`).
- **Ayanamsha** (precession) is not a body: it is encoded as a point on the
  ecliptic circle whose longitude is `ψ_t` (from `swe.get_ayanamsa_ut`) and whose
  `λ̇` is a finite-difference precession rate.

```python
from kalachakra.ephemeris import global_state
global_state.ephemeris_available()      # False until pyswisseph is installed
g = global_state.global_state_frame(jd) # (10, 7) once available
```

The encoding itself (`encode_body`) is pure and unit-tested; only the *values*
need pyswisseph.

## Installing DE441

```bash
pip install -e ".[ephemeris]"          # pyswisseph
# Download the DE441 data files (sepl_*.se1 / semo_*.se1) from the
# Swiss Ephemeris distribution and point the engine at them:
python scripts/generate_ephemeris.py --ephe-path /path/to/ephe --max-frames 10000
```

Internally the generator calls `swe.set_ephe_path(...)`, then
`swe.calc_ut(jd, body, SEFLG_SWIEPH | SEFLG_SPEED)` per body per frame, and
serializes each chunk via `kalachakra.storage.binary_store.EphemerisStore`.

## Output

Chunked `.mmap` files + a `manifest.json`, BF16 + delta-encoded. See
`docs/03_projection_and_storage.md`.
