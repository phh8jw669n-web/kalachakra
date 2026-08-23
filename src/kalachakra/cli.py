"""
Kalachakra command-line application.

Produces real, objective geometric cosmic-weather from real planetary positions.
No trained model, no synthetic data, no interpreted text — just the geometry.

Subcommands:
    reading   real weather signature for a timestamp (and optional location)
    map       real per-node potential/shear over the Earth mesh -> JSON for the UI
    scan      scan a date range for singularities (tension / shear / eclipse peaks)

Examples:
    kalachakra reading --date now --lat 51.5 --lon -0.12
    kalachakra map --date 2026-08-22T12:00 --nodes 8000 --out web/heatmap.json
    kalachakra scan --start 2026-01-01 --end 2026-12-31 --step-hours 12 --top 10
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from . import constants as C
from .analysis import weather
from .ephemeris import bodies, global_state
from .ephemeris.calendar import format_jd, parse_datetime
from .grid.geodesic import Grid, fibonacci_sphere

SIGNS = ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo", "Libra",
         "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"]


def _sign(lon_deg: float) -> str:
    lon_deg %= 360.0
    return f"{lon_deg % 30:5.2f} {SIGNS[int(lon_deg // 30)]}"


def _single_node_grid(lat_deg: float, lon_deg: float) -> Grid:
    lat, lon = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    xyz = np.array([[np.cos(lat) * np.cos(lon),
                     np.cos(lat) * np.sin(lon),
                     np.sin(lat)]])
    return Grid(xyz=xyz, lat=np.array([lat]), lon=np.array([lon]))


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------
def cmd_reading(args) -> int:
    jd = parse_datetime(args.date)
    g = global_state.global_state_frame(jd)
    lons = np.arctan2(g[:, 1], g[:, 0])
    sig = weather.frame_signature(jd, orb=args.orb)

    if args.json:
        out = {
            "timestamp_utc": format_jd(jd),
            "julian_day": jd,
            "resonance": sig.resonance,
            "tension": sig.tension,
            "net_interference": sig.net_interference,
            "potential_R": sig.potential,
            "eclipse": sig.eclipse,
            "stations": sig.stations,
            "dominant_aspects": sig.dominant_aspects,
            "positions": {
                bodies.NAMES[i]: {
                    "longitude_deg": float(np.rad2deg(lons[i]) % 360.0),
                    "speed_deg_per_day": float(np.rad2deg(g[i, 3])),
                    "retrograde": bool(g[i, 3] < 0),
                } for i in range(C.N_BODIES) if weather.BODY_WEIGHTS[i] > 0
            },
        }
        if args.lat is not None and args.lon is not None:
            grid = _single_node_grid(args.lat, args.lon)
            from .projection import spatial
            field = spatial.project(g, jd, grid)
            intensity = weather.local_intensity(field, sig.activation)[0]
            out["location"] = {"lat": args.lat, "lon": args.lon,
                               "local_intensity": float(intensity)}
        print(json.dumps(out, indent=2))
        return 0

    print(f"\n  KALACHAKRA — cosmic weather")
    print(f"  {format_jd(jd)}   (JD {jd:.5f})")
    backend = "Swiss/DE441" if global_state._MODE == "swiss" else "Moshier (analytical)"
    print(f"  ephemeris backend: {backend}\n")

    print("  planetary positions (geocentric ecliptic longitude):")
    for i in range(C.N_BODIES):
        if weather.BODY_WEIGHTS[i] == 0:
            continue
        retro = " R" if g[i, 3] < 0 else "  "
        print(f"    {bodies.NAMES[i]:8s} {_sign(np.rad2deg(lons[i])):>16s}{retro}")

    print(f"\n  harmonic resonance : {sig.resonance:6.2f}   (constructive interference)")
    print(f"  structural tension : {sig.tension:6.2f}   (destructive interference)")
    print(f"  net interference   : {sig.net_interference:+6.2f}")
    print(f"  geometric potential: {sig.potential:6.3f}   (stellium concentration R)")
    print(f"  temporal shear     : {weather.temporal_shear(jd, orb=args.orb):6.3f}   /day")

    ecl = sig.eclipse
    if ecl["is_eclipse"]:
        kind = "SOLAR" if ecl["solar_proximity"] > ecl["lunar_proximity"] else "LUNAR"
        print(f"\n  ** {kind} ECLIPSE proximity ** "
              f"(sun-moon {ecl['sun_moon_sep_deg']:.2f} deg, "
              f"moon-node {ecl['moon_node_sep_deg']:.2f} deg)")
    if sig.stations:
        print(f"\n  stationing planets : {', '.join(sig.stations)}")

    print("\n  dominant aspects (real angular geometry):")
    if not sig.dominant_aspects:
        print("    (none within orb)")
    for a in sig.dominant_aspects:
        mark = "+" if a["kind"] == "constructive" else "x"
        print(f"    [{mark}] {a['bodies'][0]:8s}-{a['bodies'][1]:8s} "
              f"{a['aspect']:11s} {a['separation_deg']:6.2f} deg  "
              f"(strength {a['strength']:.2f})")

    if args.lat is not None and args.lon is not None:
        grid = _single_node_grid(args.lat, args.lon)
        from .projection import spatial
        field = spatial.project(g, jd, grid)[0]           # (B, 5)
        intensity = weather.local_intensity(field[None], sig.activation)[0]
        print(f"\n  location {args.lat:+.3f}, {args.lon:+.3f}:")
        print(f"    local intensity  : {intensity:6.3f}")
        alt = np.rad2deg(np.arcsin(np.clip(field[:, 2], -1, 1)))
        above = [(bodies.NAMES[i], alt[i]) for i in range(C.N_BODIES)
                 if weather.BODY_WEIGHTS[i] > 0 and alt[i] > 0]
        above.sort(key=lambda t: -t[1])
        if above:
            print("    above horizon    : " +
                  ", ".join(f"{n} ({a:.0f}°)" for n, a in above))
    print()
    return 0


# --------------------------------------------------------------------------
# map
# --------------------------------------------------------------------------
def cmd_map(args) -> int:
    jd = parse_datetime(args.date)
    grid = fibonacci_sphere(args.nodes)
    print(f"Computing real weather map for {format_jd(jd)} over "
          f"{args.nodes:,} nodes...", file=sys.stderr)
    wm = weather.weather_map(jd, grid, orb=args.orb)
    payload = {
        "jd": jd,
        "timestamp_utc": format_jd(jd),
        "lat_deg": np.rad2deg(grid.lat).round(4).tolist(),
        "lon_deg": np.rad2deg(grid.lon).round(4).tolist(),
        "potential": wm["potential"].round(5).tolist(),
        "shear": wm["shear"].round(5).tolist(),
        "cluster": [-1] * grid.n_nodes,
        "summary": {
            "resonance": wm["signature"].resonance,
            "tension": wm["signature"].tension,
            "potential_R": wm["signature"].potential,
            "eclipse": wm["signature"].eclipse["is_eclipse"],
        },
    }
    with open(args.out, "w") as fh:
        json.dump(payload, fh)
    print(f"Wrote {args.out}  (potential range "
          f"[{wm['potential'].min():.3f}, {wm['potential'].max():.3f}])")
    return 0


# --------------------------------------------------------------------------
# scan
# --------------------------------------------------------------------------
def cmd_scan(args) -> int:
    jd0 = parse_datetime(args.start)
    jd1 = parse_datetime(args.end)
    step = args.step_hours / 24.0
    jds = np.arange(jd0, jd1, step)
    print(f"Scanning {len(jds):,} frames "
          f"({args.start} -> {args.end}, every {args.step_hours}h)...",
          file=sys.stderr)

    events = []
    for jd in jds:
        sig = weather.frame_signature(float(jd), orb=args.orb)
        score = sig.tension + 3.0 * max(sig.eclipse["solar_proximity"],
                                        sig.eclipse["lunar_proximity"])
        events.append((score, float(jd), sig))

    events.sort(key=lambda e: -e[0])
    print(f"\n  Top {args.top} structural-tension / singularity events:\n")
    for score, jd, sig in events[:args.top]:
        tag = ""
        if sig.eclipse["is_eclipse"]:
            tag = " [ECLIPSE]"
        top_aspect = sig.dominant_aspects[0] if sig.dominant_aspects else None
        asp = (f"  {top_aspect['bodies'][0]}-{top_aspect['bodies'][1]} "
               f"{top_aspect['aspect']}" if top_aspect else "")
        print(f"    {format_jd(jd)[:20]}  score={score:5.2f}  "
              f"tension={sig.tension:4.2f}{tag}{asp}")
    print()
    return 0


# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kalachakra", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ephe-path", default=None,
                   help="Swiss .se1 directory, DE431 (enables the full timeline)")
    p.add_argument("--jpl-file", default=None,
                   help="DE441 .bsp file for the JPL backend (full timeline)")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("reading", help="weather signature for a timestamp")
    r.add_argument("--date", default="now", help="ISO datetime or 'now' (UTC)")
    r.add_argument("--lat", type=float, default=None)
    r.add_argument("--lon", type=float, default=None)
    r.add_argument("--orb", type=float, default=weather.DEFAULT_ORB_DEG)
    r.add_argument("--json", action="store_true")
    r.set_defaults(func=cmd_reading)

    m = sub.add_parser("map", help="per-node weather map over the Earth mesh")
    m.add_argument("--date", default="now")
    m.add_argument("--nodes", type=int, default=8000)
    m.add_argument("--orb", type=float, default=weather.DEFAULT_ORB_DEG)
    m.add_argument("--out", default="web/heatmap.json")
    m.set_defaults(func=cmd_map)

    s = sub.add_parser("scan", help="scan a date range for singularities")
    s.add_argument("--start", required=True)
    s.add_argument("--end", required=True)
    s.add_argument("--step-hours", type=float, default=24.0)
    s.add_argument("--orb", type=float, default=weather.DEFAULT_ORB_DEG)
    s.add_argument("--top", type=int, default=10)
    s.set_defaults(func=cmd_scan)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not global_state.ephemeris_available():
        print("ERROR: pyswisseph not installed. Run `pip install pyswisseph`.",
              file=sys.stderr)
        return 2
    global_state.configure_from_args(ephe_path=args.ephe_path,
                                     jpl_file=args.jpl_file)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
