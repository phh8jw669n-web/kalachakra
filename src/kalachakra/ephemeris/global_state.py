"""
Global state vector G(t) in R^{10 x 7} (blueprint §2.3).

For each of the ten entities the instantaneous ecliptic state
(lambda, beta, r) and its rates (lambda_dot, beta_dot, r_dot) are read from the
Swiss Ephemeris and parameterized into a smooth, boundary-free seven-dimensional
feature vector:

    v_i(t) = [ cos(lam)cos(bet), sin(lam)cos(bet), sin(bet),
               lam_dot, bet_dot, r, r_dot ]

Stacking the ten rows yields G(t).

Ephemeris backend
-----------------
By default this uses the **Moshier** analytical ephemeris, which needs no
external data files and is valid 3000 BCE - 3000 CE (covering all of recorded
history and the present). Call :func:`configure` with ``mode="swiss"`` and a
path to the Swiss ``.se1`` files (DE431) to cover the full 10,256-year timeline
at full precision.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .. import constants as C
from . import bodies
from .bodies import Kind

try:  # pragma: no cover - depends on optional native dependency
    import swisseph as swe

    _HAS_SWE = True
except Exception:  # noqa: BLE001 - any import failure means "not available"
    swe = None  # type: ignore[assignment]
    _HAS_SWE = False


# swisseph calc flags (literals so the module imports without the native package;
# values match pyswisseph's public constants).
_FLG_JPLEPH = 1       # JPL DE441: needs the DE441 .bsp file(s)
_FLG_SWIEPH = 2       # Swiss ephemeris: needs .se1 data files
_FLG_MOSEPH = 4       # Moshier: analytical, no data files, ~3000 BCE - 3000 CE
#                      (empirical pyswisseph range: JD 625000.5 .. 2818000.5)
_FLG_SPEED = 256      # also return instantaneous speeds

_FLAG_BY_MODE = {"moshier": _FLG_MOSEPH, "swiss": _FLG_SWIEPH, "jpl": _FLG_JPLEPH}

# Active backend. Moshier by default so the system produces real data out of the
# box; switch to Swiss or JPL for the full timeline via configure().
_MODE = "moshier"


def configure(mode: str = "moshier", ephe_path: str | None = None,
              jpl_file: str | None = None) -> None:
    """Select the ephemeris backend for the full 10,256-year span.

    Parameters
    ----------
    mode : {"moshier", "swiss", "jpl"}
        "moshier" (default) needs no data files (~3000 BCE - 3000 CE). "swiss"
        uses the Swiss Ephemeris ``.se1`` files (DE431-based, ~13000 BCE-16800 CE;
        small, recommended). "jpl" reads the NASA JPL DE441 ``.bsp`` file(s).
    ephe_path : str, optional
        Directory holding the ``.se1`` files (used by "swiss"; also where a JPL
        file may live).
    jpl_file : str, optional
        Path/name of the DE441 ``.bsp`` file (required for "jpl").
    """
    global _MODE
    if mode not in _FLAG_BY_MODE:
        raise ValueError(f"mode must be one of {sorted(_FLAG_BY_MODE)}")
    if _HAS_SWE:
        if ephe_path:
            swe.set_ephe_path(ephe_path)      # type: ignore[union-attr]
        if jpl_file:
            swe.set_jpl_file(jpl_file)         # type: ignore[union-attr]
    _MODE = mode


def _calc_flags() -> int:
    return _FLAG_BY_MODE[_MODE] | _FLG_SPEED


# --- persistent configuration (written by scripts/setup_full_span.py) --------

DEFAULT_CONFIG_PATH = Path.home() / ".config" / "kalachakra" / "config.json"


def _config_search_paths() -> list[Path]:
    """Where auto_configure looks for a saved backend config, in priority order."""
    paths = []
    env_cfg = os.environ.get("KALACHAKRA_CONFIG")
    if env_cfg:
        paths.append(Path(env_cfg))
    paths.append(Path.cwd() / ".kalachakra.json")   # project-local
    paths.append(DEFAULT_CONFIG_PATH)               # user-global
    return paths


def save_config(*, mode: str = "swiss", ephe_path: str | None = None,
                jpl_file: str | None = None, path: str | Path | None = None) -> Path:
    """Persist the backend choice so future runs pick it up with no flags."""
    if mode not in _FLAG_BY_MODE:
        raise ValueError(f"mode must be one of {sorted(_FLAG_BY_MODE)}")
    target = Path(path) if path else DEFAULT_CONFIG_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    data = {"mode": mode, "ephe_path": ephe_path, "jpl_file": jpl_file}
    target.write_text(json.dumps(data, indent=2))
    return target


def configure_from_args(ephe_path: str | None = None,
                        jpl_file: str | None = None) -> str:
    """Configure from explicit flags if given, else fall back to auto_configure().

    This is the single entry point the CLI and scripts use so that, after running
    scripts/setup_full_span.py (which writes a config), every command uses the
    full-span backend with no flags — while an explicit --ephe-path/--jpl-file
    still wins.
    """
    if jpl_file:
        configure(mode="jpl", jpl_file=jpl_file, ephe_path=ephe_path)
        return "jpl"
    if ephe_path:
        configure(mode="swiss", ephe_path=ephe_path)
        return "swiss"
    return auto_configure()


def auto_configure() -> str:
    """Configure the backend from env vars / a saved config; fall back to Moshier.

    Priority:
      1. env KALACHAKRA_EPHE_PATH (swiss) or KALACHAKRA_JPL_FILE (jpl)
      2. the first config file found via :func:`_config_search_paths`
      3. Moshier (no data files)
    Returns the mode selected.
    """
    env_ephe = os.environ.get("KALACHAKRA_EPHE_PATH")
    env_jpl = os.environ.get("KALACHAKRA_JPL_FILE")
    if env_jpl:
        configure(mode="jpl", jpl_file=env_jpl)
        return "jpl"
    if env_ephe:
        configure(mode="swiss", ephe_path=env_ephe)
        return "swiss"

    for cfg_path in _config_search_paths():
        try:
            if cfg_path and cfg_path.is_file():
                data = json.loads(cfg_path.read_text())
                mode = data.get("mode", "moshier")
                configure(mode=mode, ephe_path=data.get("ephe_path"),
                          jpl_file=data.get("jpl_file"))
                return mode
        except Exception:  # noqa: BLE001 - a bad config must not crash startup
            continue

    configure(mode="moshier")
    return "moshier"


def ephemeris_available() -> bool:
    """True when pyswisseph could be imported."""
    return _HAS_SWE


def _require_swe() -> None:
    if not _HAS_SWE:
        raise RuntimeError(
            "pyswisseph is not installed. Run `pip install pyswisseph` "
            "(the default Moshier backend needs no data files)."
        )


def encode_body(lam: float, bet: float, r: float,
                lam_dot: float, bet_dot: float, r_dot: float) -> np.ndarray:
    """Pack one body's raw ecliptic state into its 7-D feature vector.

    ``lam`` and ``bet`` are radians; ``r`` is in AU; the dots are per-day rates
    (radians/day and AU/day). Pure and fully testable.
    """
    cos_b = np.cos(bet)
    return np.array(
        [
            np.cos(lam) * cos_b,
            np.sin(lam) * cos_b,
            np.sin(bet),
            lam_dot,
            bet_dot,
            r,
            r_dot,
        ],
        dtype=np.float64,
    )


def _raw_state(entity: bodies.CelestialEntity, jd_ut: float):
    """Return (lam, bet, r, lam_dot, bet_dot, r_dot) in radians / AU / per-day."""
    if entity.kind is Kind.PRECESSION:
        # Ayanamsha: a single precession angle, encoded as a point on the
        # ecliptic circle (beta = 0, r = 1) with a finite-difference rate.
        psi_deg = swe.get_ayanamsa_ut(jd_ut)  # type: ignore[union-attr]
        psi_deg_next = swe.get_ayanamsa_ut(jd_ut + 1.0)  # type: ignore[union-attr]
        lam = np.deg2rad(psi_deg)
        lam_dot = np.deg2rad(psi_deg_next - psi_deg)
        return lam, 0.0, 1.0, lam_dot, 0.0, 0.0

    # BODY or NODE: (lon, lat, dist, lon_sp, lat_sp, dist_sp).
    values, _flag = swe.calc_ut(jd_ut, entity.swe_id, _calc_flags())  # type: ignore[union-attr]
    lon_deg, lat_deg, dist_au, lon_sp, lat_sp, dist_sp = values[:6]
    lon_deg += entity.longitude_offset_deg  # Ketu = node + 180 deg
    return (
        np.deg2rad(lon_deg % 360.0),
        np.deg2rad(lat_deg),
        float(dist_au),
        np.deg2rad(lon_sp),   # deg/day -> rad/day
        np.deg2rad(lat_sp),
        float(dist_sp),
    )


def global_state_frame(jd_ut: float) -> np.ndarray:
    """Compute G(t) for a single Julian Day, shape ``(N_BODIES, 7)``."""
    _require_swe()
    rows = np.empty((C.N_BODIES, C.GLOBAL_BODY_FEATURES), dtype=np.float64)
    for i, entity in enumerate(bodies.ENTITIES):
        rows[i] = encode_body(*_raw_state(entity, jd_ut))
    return rows


def global_state_batch(jds_ut: np.ndarray) -> np.ndarray:
    """Compute G(t) for many Julian Days, shape ``(len(jds), N_BODIES, 7)``.

    pyswisseph is scalar, so this loops over frames on the CPU — the Phase-1
    generation step (§1.3) whose output is serialized to the binary store.
    """
    _require_swe()
    jds_ut = np.asarray(jds_ut, dtype=np.float64)
    out = np.empty((jds_ut.shape[0], C.N_BODIES, C.GLOBAL_BODY_FEATURES),
                   dtype=np.float64)
    for k, jd in enumerate(jds_ut):
        out[k] = global_state_frame(float(jd))
    return out


def ecliptic_longitudes(jd_ut: float) -> np.ndarray:
    """Convenience: real ecliptic longitudes (radians) of all 10 entities."""
    frame = global_state_frame(jd_ut)
    return np.arctan2(frame[:, 1], frame[:, 0])
