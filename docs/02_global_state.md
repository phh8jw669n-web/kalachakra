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
global_state.global_state_frame(jd)     # (10, 7) real positions — works today
```

`pyswisseph` is a core dependency and its default **Moshier** backend needs no
data files (valid ~1900 BCE – 4650 CE), so real `G(t)` is available out of the
box. The encoding itself (`encode_body`) is pure and unit-tested; the *values*
come from pyswisseph.

## Backends: Moshier (default) vs Swiss/DE441

```python
global_state.configure(mode="moshier")                     # default, no files
global_state.configure(mode="swiss", ephe_path="/de441")   # full 10,256-yr range
```

Moshier covers all of recorded history and the present. To reach the Kali-Yuga
epoch (3102 BCE) and the far future (past 4650 CE) — i.e. the full timeline —
download the DE441 `.se1` files from the Swiss Ephemeris distribution and pass
`--ephe-path`:

```bash
python scripts/generate_ephemeris.py --ephe-path /path/to/de441 \
    --start-date 2024-01-01 --max-frames 10000
```

Internally the generator calls `swe.calc_ut(jd, body, flags | SEFLG_SPEED)` per
body per frame and serializes each chunk via
`kalachakra.storage.binary_store.EphemerisStore`.

## Output

Chunked `.mmap` files + a `manifest.json`, BF16 + delta-encoded. See
`docs/03_projection_and_storage.md`.
