"""Structured logging + hardware telemetry for the Great Indexer (PRD page 6).

One logger fans out to a rotating file (post-run audit) and stdout (live view).
``hardware_snapshot`` interleaves memory / CPU / disk / GPU-temp metrics with the
mathematical milestone logs so an operator can spot thermal throttling or a memory
leak in real time. ``psutil`` and GPU-temp probes are optional; missing probes
degrade to ``None`` rather than crashing the run.
"""

from __future__ import annotations

import logging
import os
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

try:  # pragma: no cover - optional dependency
    import psutil
    _HAS_PSUTIL = True
except Exception:  # noqa: BLE001
    psutil = None
    _HAS_PSUTIL = False


def setup_logging(log_dir: str | Path, name: str = "great_indexer",
                  level: int = logging.INFO) -> logging.Logger:
    """A logger writing to a rotating file AND stdout with rich, timestamped lines."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    target = str((log_dir / "indexer.log").resolve())
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Idempotent within one run, but re-point the file handler if this call names a
    # different log dir (e.g. a second pipeline run in the same process) so the log
    # always lands in the active run's directory.
    already = any(isinstance(h, RotatingFileHandler)
                  and getattr(h, "baseFilename", None) == target
                  for h in logger.handlers)
    if already:
        return logger
    for h in list(logger.handlers):           # drop handlers from a prior log dir
        logger.removeHandler(h)
        h.close()

    fmt = logging.Formatter(
        "%(asctime)s.%(msecs)03d | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S")
    fh = RotatingFileHandler(target, maxBytes=32 * 1024 * 1024,
                             backupCount=5, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _gpu_temperature_c():
    """Best-effort GPU temperature in Celsius (None if no probe is available)."""
    # NVIDIA
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2)
        if out.returncode == 0 and out.stdout.strip():
            return float(out.stdout.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        pass
    # psutil thermal zones (Linux)
    if _HAS_PSUTIL and hasattr(psutil, "sensors_temperatures"):
        try:
            temps = psutil.sensors_temperatures()
            for key in ("gpu", "GPU", "coretemp", "acpitz"):
                if key in temps and temps[key]:
                    return float(temps[key][0].current)
        except Exception:  # noqa: BLE001
            pass
    return None


class DiskWriteRate:
    """Tracks bytes written to disk between calls -> MB/s (best effort)."""

    def __init__(self):
        self._t = time.time()
        self._bytes = self._read_counter()

    @staticmethod
    def _read_counter():
        if not _HAS_PSUTIL:
            return None
        try:
            io = psutil.disk_io_counters()
            return int(io.write_bytes) if io else None
        except Exception:  # noqa: BLE001
            return None

    def sample(self):
        now = time.time()
        cur = self._read_counter()
        if cur is None or self._bytes is None or now <= self._t:
            self._t, self._bytes = now, cur
            return None
        rate = (cur - self._bytes) / (now - self._t) / 1e6
        self._t, self._bytes = now, cur
        return round(rate, 2)


def hardware_snapshot(disk: DiskWriteRate | None = None) -> dict:
    """A point-in-time hardware health dict for interleaved telemetry logging."""
    snap: dict = {"psutil": _HAS_PSUTIL}
    if _HAS_PSUTIL:
        vm = psutil.virtual_memory()
        snap["mem_used_pct"] = round(vm.percent, 1)
        snap["mem_used_gb"] = round(vm.used / 1e9, 2)
        snap["mem_avail_gb"] = round(vm.available / 1e9, 2)
        try:
            snap["cpu_pct"] = round(psutil.cpu_percent(interval=None), 1)
            snap["rss_gb"] = round(psutil.Process(os.getpid()).memory_info().rss / 1e9, 2)
        except Exception:  # noqa: BLE001
            pass
    gpu = _gpu_temperature_c()
    if gpu is not None:
        snap["gpu_temp_c"] = gpu
    if disk is not None:
        rate = disk.sample()
        if rate is not None:
            snap["disk_write_mbps"] = rate
    return snap


def format_hw(snap: dict) -> str:
    """Compact one-line rendering of a hardware snapshot for logs."""
    parts = []
    if "mem_used_pct" in snap:
        parts.append(f"mem={snap['mem_used_pct']}% ({snap['mem_used_gb']}GB used, "
                     f"{snap['mem_avail_gb']}GB free)")
    if "rss_gb" in snap:
        parts.append(f"rss={snap['rss_gb']}GB")
    if "cpu_pct" in snap:
        parts.append(f"cpu={snap['cpu_pct']}%")
    if "gpu_temp_c" in snap:
        parts.append(f"gpu={snap['gpu_temp_c']}C")
    if "disk_write_mbps" in snap:
        parts.append(f"disk_w={snap['disk_write_mbps']}MB/s")
    return "  ".join(parts) if parts else "(no hardware probes available)"
