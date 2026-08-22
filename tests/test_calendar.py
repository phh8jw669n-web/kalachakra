import numpy as np

from kalachakra.ephemeris import calendar as cal


def test_j2000_reference():
    # 2000-01-01 12:00 UTC is exactly JD 2451545.0.
    assert cal.gregorian_to_jd(2000, 1, 1, 12, 0, 0) == 2451545.0


def test_roundtrip_gregorian():
    for jd in [2451545.0, 2460409.26181, 2299161.0, 1721426.0]:
        y, mo, d, h, mi, s = cal.jd_to_gregorian(jd)
        back = cal.gregorian_to_jd(y, mo, d, h, mi, s)
        assert abs(back - jd) < 1e-6, (jd, back)


def test_parse_datetime_iso_and_z():
    a = cal.parse_datetime("2024-04-08T18:17:00Z")
    b = cal.parse_datetime("2024-04-08T18:17:00+00:00")
    assert abs(a - b) < 1e-9
    assert abs(a - 2460409.26181) < 1e-4


def test_parse_date_only_defaults_to_noon_utc():
    jd = cal.parse_datetime("2024-04-08")
    # midnight UTC -> JD .5 boundary
    assert abs((jd + 0.5) - round(jd + 0.5)) < 1e-6


def test_format_jd_roundtrips_reasonably():
    s = cal.format_jd(2451545.0)
    assert "2000-01-01" in s and "CE" in s


def test_bce_year_numbering():
    # Astronomical year 0 == 1 BCE.
    s = cal.format_jd(cal.gregorian_to_jd(0, 1, 1, 12))
    assert "BCE" in s
