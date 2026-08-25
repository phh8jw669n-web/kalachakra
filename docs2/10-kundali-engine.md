# 10 · The Kundali Twin Search Engine

`kundali/` — a standalone sidereal (Vedic/Jyotish) astrology search, independent of
the neural networks and the mesh. Given a birth chart, it sweeps history for days
whose sidereal geometry matches at eight escalating tiers, and — for
house‑dependent tiers — solves the `(lat, lon)` curve on Earth where the match
occurs.

Files: `kundali/{astro,db,search}.py`, `scripts/{build_kundali_db,serve_kundali}.py`,
`web/kundali.html`.

---

## 1. `astro.py` — sidereal math (source of truth)

All longitudes are **sidereal** using the **Lahiri ayanamsha** (`configure()` sets
`SIDM_LAHIRI`). Uses Swiss Ephemeris via `ephemeris.global_state`.

**Constants.** `SIGNS` (12, Aries…Pisces), `NAKSHATRAS` (27), `BODIES` — the nine
grahas `(sun, moon, mars, mercury, jupiter, venus, saturn, rahu, ketu)` with swe ids
& offsets (Ketu = Rahu + 180°), `SLOW_BODIES = (saturn, jupiter, rahu, ketu)`.

**Divisions.**

| Function | Meaning |
|---|---|
| `sign_of(lon)` | zodiac sign 0–11 (`lon//30`) |
| `degree_in_sign(lon)` | degrees within sign `[0,30)` |
| `nakshatra_of(lon)` | nakshatra 0–26 (arc = 13°20′) |
| `navamsa_sign(lon)` | D9 navamsa sign via the element rule `(sign·9 + idx) % 12` |
| `body_longitude(jd, swe_id, offset)` / `body_longitudes(jd)` | sidereal longitude(s) |
| `ayanamsha(jd)` | Lahiri ayanamsha (deg) |

**Ascendant (Lagna).**

- `ascendant_sidereal_deg(jd, lat, lon)` = tropical Ascendant (from RAMC = GMST+lon,
  obliquity, latitude) − ayanamsha.
- `ascendant_sign(jd, lat, lon)`.
- `ascendant_signs_over_longitudes(jd, lat, lons)` — **vectorized**: obliquity/GMST/
  ayanamsha computed once, RAMC varied over the longitude grid → rising sign per
  longitude (fast enough for 1000+ dates × a latitude fan).
- `longitudes_for_ascendant_sign(jd, lat, target_sign, coarse_step=0.5, refine=False)`
  — the band centers of longitude that put the Lagna in `target_sign` (widest first;
  `refine` narrows toward arc‑minute for the divisional tier). `_bands_from_hit`
  handles the ±180° seam via circular mean.
- `globe_ascendant_points(jd, target_sign, lats_deg=GLOBE_LATS, coarse_step=0.5)` —
  the **twin locus**: one `{lat, lon}` per latitude in the fan, so a fixed rising
  sign traces a lat/lon curve **across the globe** (one broadcast solve).

**`natal_chart(jd, lat, lon)`** — the full search key: per body `{lon, sign, deg,
nak, nav, house}`, the Ascendant sign, `conjunctions` (bodies sharing a sign, degree‑
ordered), and `houses` (house → resident bodies). Houses are whole‑sign:
`house = (sign − asc_sign) % 12 + 1`.

> **Why latitude matters (and mostly doesn't).** Planetary signs are geocentric —
> identical everywhere on Earth at an instant. Only the Ascendant/houses depend on
> location. So sign‑based tiers are *location‑free*; house‑based tiers trace a
> lat/lon curve (`globe_ascendant_points`).

---

## 2. `db.py` — the daily sidereal DuckDB

A lightweight, indexed daily ephemeris table for millisecond tier sweeps.

- `build_arrays(start_jd, end_jd, step_days=1.0)` — sample **integer JDs (noon UT)**
  (`np.ceil(start − 1e‑9)`), compute per‑body `sign/deg/nak/nav/lon` columns + `year`.
- `write_duckdb(out_path, cols, meta)` — a wide `positions` table with **indexed sign
  columns** + a `meta` table (start/end year, ayanamsha).
- `build_db(out_path, start_jd, end_jd, step_days=1.0)` — end to end.
  Built by `scripts/build_kundali_db.py --start-year --end-year --out k.duckdb`.

Row schema (per body `b`): `b_sign, b_deg, b_nak, b_nav, b_lon` for the nine grahas,
plus `jd` and `year`.

---

## 3. `search.py` — the eight tiers

`KundaliSearch(db_path)`; `set_natal(jd, lat, lon)` builds the natal chart key.

**Tiers (broad → rare):**

| Tier | Name | Criterion | House‑dependent |
|---|---|---|---|
| 1 | Generational Twin | slow bodies (Sat/Jup/Rahu/Ketu) share signs | no |
| 2 | Psychological Twin | all nine bodies share signs | no |
| 3 | Geographic Mirror | same 1–12 house sequence (signs free) → solve longitude | **yes** |
| 4 | Core Ascendant Lock | all signs match AND Lagna locked to natal sign | **yes** |
| 5 | Nakshatra Twin | tier 4 + Moon in the same nakshatra | no* |
| 6 | Planetary War Lock | tier 5 + conjunct planets keep degree order | no* |
| 7 | Absolute Twin | every body within a 5° orb of the natal degree | no |
| 8 | Divisional Lock | D9 navamsa identical + tier‑7 orb + refined longitude | **yes** |

Each tier is **one indexed SQL sweep** (`_tier_where`) over `positions`, built from
per‑body WHERE clauses (`_signs_where`, `_mirror_where`, `_war_where`, `_orb_where`,
`_nav_where`; `_circ` for circular longitude distance).

**Dynamic constraints.** `resolve_active(active_planets, active_houses)` selects which
bodies are enforced — a planet subset ("Planetary Tethers") or the natal residents of
chosen houses ("Selective House Locking"). `match_score = |active| / 9`.

**Date range.** `search(..., start_year, end_year)` adds `AND year ≥ s AND year ≤ e`
(also honored by `counts_by_tier`).

**`search(tier, limit=300, active_planets, active_houses, start_year, end_year) →`**

```json
{ "results": [ {jd, date, year, time_utc, latitude, total_matched,
                longitude, longitudes, ascendant_sign, local_time,
                globe_points:[{lat,lon}…], longitude_constrained} … ],
  "active_planets": […], "active_constraint_count": N, "match_score": N/9,
  "location_free": bool }
```

- Sign tiers set `longitude_constrained=false` and `location_free=true` (twin exists
  anywhere on Earth that day). House tiers solve `longitude`/`longitudes`, add
  `local_time` (local mean time at the solved longitude) and, for the leading results
  (`_GLOBE_MAX=200`), a `globe_points` latitude fan.
- Each result carries the exact `time_utc` (the noon‑UT sample instant) alongside the
  date; the UI supports up to `limit=1000`.
- `counts_by_tier(...)` — cheap per‑tier existence probe for the availability ticks.

---

## 4. Dashboard (`serve_kundali.py` + `web/kundali.html`)

Endpoints in [08 §7.1](08-api-and-serving.md). The 2‑D equirectangular map:

- Draws result pins; **house tiers** draw the blue **twin locus** spanning latitudes
  (`globe_points`), while **sign tiers** tint the whole map ("location‑free — same
  signs worldwide on each date").
- Controls: birth date/time/tz/lat/lon + presets; an 8‑tier slider with availability
  ticks; **By Planets / By Houses** constraint toggles; an optional **year range**;
  field resolution / list limit (1000). Clicking a pin shows date + `time_utc` +
  longitude + `local_time`.

Sidereal math is never approximated for UI convenience: signs, noon‑UT sampling, and
tier logic are exact; the globe locus is the same Ascendant equation evaluated over a
latitude grid.

---

## 5. Build & run

```bash
pip install -e ".[index,serve]"        # duckdb + fastapi
python scripts/build_kundali_db.py --start-year 1800 --end-year 2100 --out kundali.duckdb
python scripts/serve_kundali.py --db kundali.duckdb     # then open the dashboard
```

The full 10,256‑year DB needs the Swiss `.se1` files (deep time is outside Moshier);
a modern range works file‑free. Tested in `tests/test_kundali.py`.
