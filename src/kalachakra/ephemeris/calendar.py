"""
Civil time <-> Julian Day conversion (proleptic Gregorian, UTC).

Pure-Python implementation of the standard Fliegel-Van Flandern algorithm so the
CLI and analysis layers can turn a human date/time into the ``jd_ut`` that the
ephemeris engine expects — with no dependency on pyswisseph. Valid across the
entire 10,256-year timeline.
"""

from __future__ import annotations

from datetime import datetime, timezone


def gregorian_to_jd(year: int, month: int, day: int,
                    hour: float = 12.0, minute: float = 0.0,
                    second: float = 0.0) -> float:
    """Proleptic-Gregorian calendar date (UTC) -> Julian Day.

    ``year`` uses astronomical numbering (1 BCE == year 0, 2 BCE == -1, ...).
    """
    a = (14 - month) // 12
    y = year + 4800 - a
    m = month + 12 * a - 3
    jdn = (day + (153 * m + 2) // 5 + 365 * y + y // 4
           - y // 100 + y // 400 - 32045)
    day_fraction = (hour - 12.0) / 24.0 + minute / 1440.0 + second / 86400.0
    return jdn + day_fraction


def jd_to_gregorian(jd: float) -> tuple[int, int, int, int, int, float]:
    """Julian Day -> (year, month, day, hour, minute, second) in UTC."""
    jd_plus = jd + 0.5
    z = int(jd_plus)
    frac = jd_plus - z

    a = z + 32044
    b = (4 * a + 3) // 146097
    c = a - (146097 * b) // 4
    d = (4 * c + 3) // 1461
    e = c - (1461 * d) // 4
    m = (5 * e + 2) // 153

    day = e - (153 * m + 2) // 5 + 1
    month = m + 3 - 12 * (m // 10)
    year = 100 * b + d - 4800 + m // 10

    seconds_total = frac * 86400.0
    hour = int(seconds_total // 3600)
    minute = int((seconds_total % 3600) // 60)
    second = seconds_total % 60.0
    return year, month, day, hour, minute, second


def datetime_to_jd(dt: datetime) -> float:
    """timezone-aware (or naive-as-UTC) :class:`datetime` -> Julian Day."""
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc)
    return gregorian_to_jd(dt.year, dt.month, dt.day,
                           dt.hour, dt.minute, dt.second + dt.microsecond / 1e6)


def parse_datetime(text: str) -> float:
    """Parse an ISO-8601 string (or 'now') into a Julian Day (UTC).

    Accepts e.g. ``2026-08-22``, ``2026-08-22T14:30``, ``2026-08-22T14:30:00Z``.
    """
    if text.strip().lower() == "now":
        return datetime_to_jd(datetime.now(timezone.utc))
    t = text.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(t)
    except ValueError as exc:
        raise ValueError(f"could not parse datetime {text!r}: {exc}") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return datetime_to_jd(dt)


def format_jd(jd: float) -> str:
    """Human-readable UTC timestamp for a Julian Day."""
    y, mo, d, h, mi, s = jd_to_gregorian(jd)
    era = "CE" if y > 0 else "BCE"
    yy = y if y > 0 else 1 - y
    return f"{yy:04d}-{mo:02d}-{d:02d} {h:02d}:{mi:02d}:{s:04.1f} UTC {era}"
