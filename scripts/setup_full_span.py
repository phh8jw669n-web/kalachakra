#!/usr/bin/env python3
"""
One-command setup for the FULL 10,256-year span.

Downloads exactly the Swiss Ephemeris data files needed to cover Kalachakra's
3102 BCE -> 7154 CE timeline, verifies them against the ephemeris engine, and
writes a config file so every other command (kalachakra, train.py, analyze.py,
generate_ephemeris.py) uses the full-span backend automatically — no flags.

Typical use (on a machine with internet access):

    pip install -e ".[train]"
    python scripts/setup_full_span.py         # download + verify + configure
    python scripts/train.py --store data/full # now trains over the full span

Options:
    --dest DIR         where to store the .se1 files (default ~/.kalachakra/ephe)
    --start-year / --end-year   astronomical years to cover (default full span)
    --mirror-all       fetch every sepl*/semo* block, not just the needed subset
    --jpl              download raw JPL DE441 .bsp kernels instead of .se1
    --base-url URL     override the .se1 download directory
    --config-path P    where to write the config (default ./.kalachakra.json)
    --configure-only   don't download; just verify + configure an existing --dest
                       (use this if you downloaded the files manually)
    --dry-run          list what would be downloaded, then exit
    --force            re-download files that already exist
    --no-verify        skip the pyswisseph verification step
    --no-config        do not write a config file

Files are fetched from the official Swiss Ephemeris GitHub mirror by default.
The .se1 route is ~40 MB and covers the whole span (DE431). --jpl fetches the
literal DE441 kernels (~3 GB). See instructions.txt for details.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kalachakra.ephemeris import se1_files                        # noqa: E402

# Default download source: the official Swiss Ephemeris GitHub mirror
# (aloistr/swisseph). Verified to host the complete sepl*/semo* .se1 set and to
# be reliably reachable. The astro.com FTP path is offered as an alternative via
# --base-url, but it has been observed to 404 on some files/networks.
DEFAULT_BASE_URL = "https://raw.githubusercontent.com/aloistr/swisseph/master/ephe"
ASTRO_BASE_URL = "https://www.astro.com/ftp/swisseph/ephe"
JPL_BASE_URL = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets"
JPL_FILES = ("de441_part-1.bsp", "de441_part-2.bsp")
_UA = {"User-Agent": "kalachakra-setup/1.0 (+https://github.com/)"}


def build_se1_plan(dest: Path, base_url: str, start_year: int, end_year: int,
                   mirror_all: bool) -> list[tuple[str, Path]]:
    """Return [(url, local_path)] for the .se1 files to fetch."""
    base_url = base_url.rstrip("/")
    if mirror_all:
        blocks = se1_files.all_blocks()
    else:
        blocks = se1_files.blocks_for_years(start_year, end_year)
    plan = []
    for b in blocks:
        for prefix in se1_files.DEFAULT_PREFIXES:
            name = b.filename(prefix)
            plan.append((f"{base_url}/{name}", dest / name))
    return plan


def build_jpl_plan(dest: Path) -> list[tuple[str, Path]]:
    return [(f"{JPL_BASE_URL}/{name}", dest / name) for name in JPL_FILES]


def download_file(url: str, dest: Path, *, force: bool = False,
                  retries: int = 4, min_bytes: int = 1024) -> bool:
    """Download ``url`` to ``dest`` atomically, with retries. Returns True if OK."""
    if dest.exists() and dest.stat().st_size >= min_bytes and not force:
        print(f"  skip (exists): {dest.name}")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=_UA)
            with urllib.request.urlopen(req, timeout=60) as resp:
                with tempfile.NamedTemporaryFile(delete=False,
                                                 dir=dest.parent) as tmp:
                    shutil.copyfileobj(resp, tmp)
                    tmp_path = Path(tmp.name)
            size = tmp_path.stat().st_size
            head = tmp_path.read_bytes()[:64].lstrip()
            if size < min_bytes or head[:1] in (b"<",):
                tmp_path.unlink(missing_ok=True)
                raise OSError(f"suspicious download ({size} bytes) — not a data file")
            tmp_path.replace(dest)
            print(f"  ok: {dest.name} ({size/1e6:.1f} MB)")
            return True
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt < retries:
                wait = 2 ** attempt
                print(f"  retry {attempt}/{retries} for {dest.name} "
                      f"in {wait}s ({exc})")
                time.sleep(wait)
    print(f"  FAILED: {dest.name} ({last_err})", file=sys.stderr)
    return False


def verify_swiss(dest: Path) -> bool:
    """Confirm the ephemeris can compute the Kali Yuga epoch from ``dest``."""
    from kalachakra.ephemeris import global_state
    if not global_state.ephemeris_available():
        print("  pyswisseph not installed — skipping verification.", file=sys.stderr)
        return False
    global_state.configure(mode="swiss", ephe_path=str(dest))
    frame = global_state.global_state_frame(588465.5)  # 3102 BCE Kali Yuga epoch
    ok = frame.shape == (10, 7)
    print(f"  verified: computed G(t) at the Kali Yuga epoch, shape {frame.shape}")
    return ok


def verify_jpl(part1: Path) -> bool:
    from kalachakra.ephemeris import global_state
    if not global_state.ephemeris_available():
        return False
    global_state.configure(mode="jpl", jpl_file=str(part1))
    frame = global_state.global_state_frame(2451545.0)  # J2000, in part-1 range
    print(f"  verified: computed G(t) via JPL, shape {frame.shape}")
    return frame.shape == (10, 7)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dest", type=Path, default=Path.home() / ".kalachakra" / "ephe")
    p.add_argument("--start-year", type=int, default=se1_files.KALACHAKRA_START_YEAR,
                   help="astronomical start year (default 3102 BCE == -3101)")
    p.add_argument("--end-year", type=int, default=se1_files.KALACHAKRA_END_YEAR)
    p.add_argument("--mirror-all", action="store_true")
    p.add_argument("--jpl", action="store_true", help="download DE441 .bsp instead")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--config-path", type=Path, default=Path(".kalachakra.json"))
    p.add_argument("--configure-only", action="store_true",
                   help="skip download; verify + configure an existing --dest folder")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--no-verify", action="store_true")
    p.add_argument("--no-config", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    dest = args.dest.expanduser()

    if args.jpl:
        plan = build_jpl_plan(dest)
        kind = "JPL DE441 .bsp"
    else:
        plan = build_se1_plan(dest, args.base_url, args.start_year,
                              args.end_year, args.mirror_all)
        kind = "Swiss .se1 (DE431)"

    print(f"Kalachakra full-span setup — {kind}")
    print(f"  span : {se1_files.fmt_year(args.start_year)} .. "
          f"{se1_files.fmt_year(args.end_year)}")
    print(f"  dest : {dest}")
    print(f"  files: {len(plan)}")
    if not args.jpl and not args.mirror_all:
        for b in se1_files.blocks_for_years(args.start_year, args.end_year):
            print(f"    {b.tag}  {se1_files.fmt_year(b.start_year):>9s} .. "
                  f"{se1_files.fmt_year(b.end_year):>9s}")

    if args.dry_run:
        print("\n[dry run] would download:")
        for url, path in plan:
            print(f"    {url} -> {path}")
        return 0

    if args.configure_only:
        # Manual-download path: the user placed the files in --dest themselves.
        missing = [path.name for _url, path in plan if not path.exists()]
        if missing:
            print(f"\n{len(missing)} expected file(s) missing from {dest}:",
                  file=sys.stderr)
            for name in missing[:8]:
                print(f"    {name}", file=sys.stderr)
            if len(missing) > 8:
                print(f"    ... and {len(missing) - 8} more", file=sys.stderr)
            print("Download them (see instructions.txt) or drop --configure-only.",
                  file=sys.stderr)
            return 1
        print(f"\nUsing {len(plan)} existing files in {dest} (no download).")
    else:
        print(f"\nDownloading from {args.base_url} ...")
        ok = all(download_file(url, path, force=args.force) for url, path in plan)
        if not ok:
            print("\nSome files failed to download. Re-run to resume (existing "
                  "files are skipped), try a different --base-url (e.g. "
                  f"{ASTRO_BASE_URL}), or download manually and use "
                  "--configure-only.", file=sys.stderr)
            return 1

    if not args.no_verify:
        print("\nVerifying...")
        verified = verify_jpl(dest / JPL_FILES[0]) if args.jpl else verify_swiss(dest)
        if not verified:
            print("Verification did not pass; see messages above.", file=sys.stderr)

    if not args.no_config:
        from kalachakra.ephemeris import global_state
        if args.jpl:
            cfg = global_state.save_config(mode="jpl",
                                           jpl_file=str(dest / JPL_FILES[0]),
                                           path=args.config_path)
        else:
            cfg = global_state.save_config(mode="swiss", ephe_path=str(dest),
                                           path=args.config_path)
        print(f"\nWrote config: {cfg}")
        print("  (auto-loaded by the CLI and scripts when run from this directory)")

    print("\nDone. The project is ready for the full span. Next:")
    print("  kalachakra reading --date -3101-02-18      # Kali Yuga epoch, real")
    print("  python scripts/generate_ephemeris.py --out data/full --start-frame 0 \\")
    print("      --max-frames 1000000 --chunk-frames 1000000")
    print("  python scripts/train.py --store data/full  # trains over the full span")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
