"""
Project Kalachakra — canonical constants.

This module is the single source of truth for every fixed quantity described in
the architecture blueprint (sections 1-7). Nothing here is a "magic number":
each value is annotated with its provenance and, where the blueprint states a
derived figure (frame count, storage footprint, throughput), the derivation is
reproduced so the number can be checked rather than trusted.

Only the Python standard library is imported so this module is safe to import
in any environment (no numpy / torch / swisseph required).
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# 2.1  Astronomical bounds and ephemeris engine
# ---------------------------------------------------------------------------

#: Julian Day of the Kali Yuga epoch — 3102-02-18 BCE, 00:00:00 UTC.
#: This is the closed-loop training anchor (blueprint §2.1).
KALI_YUGA_EPOCH_JD: float = 588465.5

#: Length of the simulated timeline in tropical years (blueprint §2.1).
TIMELINE_YEARS: int = 10_256

#: Terminal calendar year of the simulation (proleptic Gregorian, blueprint §2.1).
TIMELINE_END_YEAR_CE: int = 7154

#: JPL integration dataset wrapped by pyswisseph. DE441 is the widest window of
#: provably zero-error celestial mechanics and encloses the whole timeline.
EPHEMERIS_DATASET: str = "DE441"

# ---------------------------------------------------------------------------
# 2.2  Native, non-arbitrary measurement units
# ---------------------------------------------------------------------------

#: Temporal sampling step. The Vighatika is a native harmonic == 24 SI seconds
#: (blueprint §2.2). Sampling at this rate advances the local eastern horizon by
#: ~0.1 deg per frame, aligning ingestion with Earth's physical rotation.
VIGHATIKA_SECONDS: int = 24

#: Sidereal day used to relate the Vighatika step to horizon advance (seconds).
SIDEREAL_DAY_SECONDS: float = 86_164.0905

#: Horizon advance per frame, in degrees. Derived, not assumed:
#:     360 deg / SIDEREAL_DAY_SECONDS * VIGHATIKA_SECONDS ~= 0.10027 deg.
HORIZON_ADVANCE_DEG_PER_FRAME: float = 360.0 / SIDEREAL_DAY_SECONDS * VIGHATIKA_SECONDS

#: Number of uniformly distributed observer nodes on the Earth mesh
#: (Level-5 hierarchical spatial index, blueprint §2.2).
N_SPATIAL_NODES: int = 122_880

#: Mean Earth radius in astronomical units (6371.0088 km / 149597870.7 km/AU).
#: Sets the observer's offset from geocentre for the topocentric parallax
#: correction in the projection (§3.1). A spherical Earth is used — the ~0.3 %
#: polar flattening is far below the field's other approximations.
EARTH_RADIUS_AU: float = 6_371.0088 / 149_597_870.7

#: Version of the G(t) -> E(t,s) projection semantics. Bump whenever the local
#: field's meaning changes so trained models and built indexes can refuse to mix
#: incompatible artifacts. History:
#:   1 = geocentric directions only (pre-parallax)
#:   2 = topocentric, parallax applied to physical bodies only (nodes/Ayanamsha
#:       kept geocentric)
PROJECTION_VERSION: int = 2

# ---------------------------------------------------------------------------
# 2.3  Global state vector definition
# ---------------------------------------------------------------------------

#: Ten active celestial entities tracked by the global state G(t) (blueprint §2.3).
#: Seven observable masses + the two lunar nodes + the precession (Ayanamsha) vector.
N_BODIES: int = 10

#: Per-body feature width of the *global* state vector v_i(t) (blueprint §2.3):
#:   [cosλcosβ, sinλcosβ, sinβ, λ̇, β̇, r, ṙ]
GLOBAL_BODY_FEATURES: int = 7

#: Per-body feature width of the *local* projected field e_i(s,t) (blueprint §3.1):
#:   [cosθcosh, sinθcosh, sinh, cosΔφ, sinΔφ]
LOCAL_BODY_FEATURES: int = 5

# Convenience: flattened widths.
GLOBAL_STATE_WIDTH: int = N_BODIES * GLOBAL_BODY_FEATURES   # 70
LOCAL_FIELD_WIDTH: int = N_BODIES * LOCAL_BODY_FEATURES     # 50

# ---------------------------------------------------------------------------
# 4.3  Latent bottleneck
# ---------------------------------------------------------------------------

#: Dimensionality of the continuous latent manifold z(t, s) (blueprint §4.3).
LATENT_DIM: int = 64

# ---------------------------------------------------------------------------
# 1.4  Hardware / memory budget (Apple Silicon M4 Max, zero-cloud budget)
# ---------------------------------------------------------------------------

UNIFIED_MEMORY_GB: int = 128
MEMORY_BANDWIDTH_GB_S: int = 546

# Strict partition of unified memory (must sum to UNIFIED_MEMORY_GB).
MPS_TENSOR_BUDGET_GB: int = 80          # PyTorch MPS: weights, gradients, optimizer states
RING_BUFFER_BUDGET_GB: int = 20         # CPU async prefetch of ephemeris chunks
TESTING_DAEMON_BUDGET_GB: int = 20      # parallel benchmark evaluator
SYSTEM_OVERHEAD_GB: int = 8             # macOS overhead

#: Sustained training throughput target (blueprint §1.4).
TARGET_FRAMES_PER_SECOND: int = 150_000

#: Expected wall-clock training duration in days (blueprint §1.4).
TRAINING_DURATION_DAYS: int = 90

# ---------------------------------------------------------------------------
# 3.2  Storage layout
# ---------------------------------------------------------------------------

#: Bytes per BF16 scalar.
BF16_BYTES: int = 2

#: On-disk footprint of the compressed, delta-encoded ephemeris matrix
#: (blueprint §3.2), in gigabytes.
STORAGE_FOOTPRINT_GB: int = 300

#: File extension used for the memory-mapped binary chunk files.
BINARY_CHUNK_EXTENSION: str = ".mmap"


# ---------------------------------------------------------------------------
# Derived quantities (recomputed here so the blueprint's figures are auditable)
# ---------------------------------------------------------------------------

#: Julian (astronomical) year length in days, used for frame-count derivation.
DAYS_PER_YEAR: float = 365.25
SECONDS_PER_DAY: int = 86_400


def total_temporal_frames() -> int:
    """Number of 24-second frames spanning the full 10,256-year timeline.

    Derivation (blueprint §3.1 states ~13.4 billion):
        TIMELINE_YEARS * DAYS_PER_YEAR * SECONDS_PER_DAY / VIGHATIKA_SECONDS
    """
    seconds = TIMELINE_YEARS * DAYS_PER_YEAR * SECONDS_PER_DAY
    return int(seconds // VIGHATIKA_SECONDS)


def raw_uncompressed_bytes() -> int:
    """Raw (uncompressed) BF16 footprint of the global-state timeline in bytes.

    G(t) has GLOBAL_STATE_WIDTH scalars per frame; each is 2 bytes in BF16.
    The blueprint compresses this (delta encoding) to ~STORAGE_FOOTPRINT_GB.
    """
    return total_temporal_frames() * GLOBAL_STATE_WIDTH * BF16_BYTES


def memory_partition_gb() -> dict[str, int]:
    """The unified-memory partition; the values must sum to UNIFIED_MEMORY_GB."""
    return {
        "mps_tensors": MPS_TENSOR_BUDGET_GB,
        "ring_buffer": RING_BUFFER_BUDGET_GB,
        "testing_daemon": TESTING_DAEMON_BUDGET_GB,
        "system_overhead": SYSTEM_OVERHEAD_GB,
    }


@dataclass(frozen=True)
class TimelineBounds:
    """Julian-Day span of the simulation, resolved from the epoch anchor."""

    start_jd: float
    end_jd: float
    n_frames: int

    @property
    def span_days(self) -> float:
        return self.end_jd - self.start_jd


def timeline_bounds() -> TimelineBounds:
    """Resolve the [start, end] Julian-Day window and its frame count."""
    n_frames = total_temporal_frames()
    span_days = TIMELINE_YEARS * DAYS_PER_YEAR
    return TimelineBounds(
        start_jd=KALI_YUGA_EPOCH_JD,
        end_jd=KALI_YUGA_EPOCH_JD + span_days,
        n_frames=n_frames,
    )


def _validate() -> None:
    """Internal consistency checks; run at import time and by the test-suite."""
    partition = memory_partition_gb()
    assert sum(partition.values()) == UNIFIED_MEMORY_GB, \
        "memory partition must sum to unified memory (80+20+20+8 == 128)"
    assert GLOBAL_STATE_WIDTH == 70
    assert LOCAL_FIELD_WIDTH == 50
    # Horizon advance is ~0.1 deg/frame (blueprint §2.2).
    assert abs(HORIZON_ADVANCE_DEG_PER_FRAME - 0.1) < 1e-3


_validate()
