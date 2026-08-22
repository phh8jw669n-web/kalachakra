"""
Parallel testing daemon (blueprint §5.3).

Runs asynchronously from the GPU training loop. It watches the checkpoint
directory for newly written *era* snapshots, loads each into a separate CPU
memory space, evaluates the encoder against a suite of known celestial benchmark
events (total solar eclipses, great conjunctions, synthetic resonance cases) and
logs a Resonance Divergence Metric — never pausing the primary training stream.

The benchmark evaluation reuses the same numpy projection + geodesic loss so it
has no dependence on the training device.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..ephemeris import global_state
from ..grid.geodesic import Grid, default_grid
from ..losses import reference as ref_loss
from ..projection import spatial


@dataclass(frozen=True)
class BenchmarkEvent:
    """A historical or synthetic event with a known Julian Day."""

    name: str
    jd_ut: float
    kind: str  # "eclipse" | "conjunction" | "synthetic"


#: A small default suite. Julian Days are illustrative anchors; extend as needed.
DEFAULT_BENCHMARKS: tuple[BenchmarkEvent, ...] = (
    BenchmarkEvent("total_solar_eclipse_-1223", 1_213_073.5, "eclipse"),
    BenchmarkEvent("great_conjunction_2020", 2_459_204.0, "conjunction"),
    BenchmarkEvent("synthetic_grand_trine", 2_451_545.0, "synthetic"),
)


@dataclass
class DaemonConfig:
    checkpoint_dir: Path
    log_path: Path
    poll_seconds: float = 30.0
    era_glob: str = "era_*.pt"
    benchmarks: tuple[BenchmarkEvent, ...] = DEFAULT_BENCHMARKS


def resonance_divergence(model_encode, grid: Grid,
                         benchmarks=DEFAULT_BENCHMARKS) -> dict[str, float]:
    """Score how well ``model_encode`` reproduces benchmark field structure.

    ``model_encode`` maps a projected field E (numpy ``(T, N, B*5)``) to a latent
    code and back to a reconstruction of the same shape. The metric is the mean
    composite geodesic loss across benchmarks; lower means the autoencoder has
    mastered the geometry. Requires pyswisseph to build the ground-truth field.
    """
    if not global_state.ephemeris_available():
        raise RuntimeError("pyswisseph required to evaluate benchmarks")

    scores: dict[str, float] = {}
    for ev in benchmarks:
        g = global_state.global_state_frame(ev.jd_ut)          # (B, 7)
        field = spatial.project(g, ev.jd_ut, grid)             # (N, B, 5)
        target = field[None]                                   # (1, N, B, 5) as T=1
        flat = target.reshape(1, grid.n_nodes, -1)
        recon_flat = model_encode(flat)                        # (1, N, B*5)
        recon = recon_flat.reshape(1, grid.n_nodes, -1, 5)
        lons, _ = spatial.decode_ecliptic(g)
        parts = ref_loss.composite_loss(
            recon, target,
            recon_lons=lons[None], target_lons=lons[None],
        )
        scores[ev.name] = parts["total"]
    scores["mean_divergence"] = float(np.mean(list(scores.values())))
    return scores


class TestingDaemon:
    """File-watching loop that scores each new era checkpoint."""

    def __init__(self, cfg: DaemonConfig, load_encoder):
        """``load_encoder(path) -> model_encode`` builds the eval callable."""
        self.cfg = cfg
        self.load_encoder = load_encoder
        self.grid = default_grid()
        self._seen: set[str] = set()
        cfg.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _new_checkpoints(self) -> list[Path]:
        found = sorted(self.cfg.checkpoint_dir.glob(self.cfg.era_glob))
        fresh = [p for p in found if p.name not in self._seen]
        self._seen.update(p.name for p in fresh)
        return fresh

    def _log(self, record: dict) -> None:
        with self.cfg.log_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")

    def poll_once(self) -> list[dict]:
        records = []
        for ckpt in self._new_checkpoints():
            encode = self.load_encoder(ckpt)
            scores = resonance_divergence(encode, self.grid, self.cfg.benchmarks)
            record = {"checkpoint": ckpt.name, "t": time.time(), **scores}
            self._log(record)
            records.append(record)
        return records

    def run(self, stop_after: float | None = None) -> None:
        start = time.monotonic()
        while True:
            self.poll_once()
            if stop_after is not None and time.monotonic() - start >= stop_after:
                return
            time.sleep(self.cfg.poll_seconds)
