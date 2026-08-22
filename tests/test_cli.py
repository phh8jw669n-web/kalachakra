"""CLI smoke tests on real data. Skipped if pyswisseph is absent."""

import json

import pytest

from kalachakra import cli
from kalachakra.ephemeris import global_state

pytestmark = pytest.mark.skipif(
    not global_state.ephemeris_available(),
    reason="pyswisseph not installed",
)


def test_reading_json_reports_real_eclipse(capsys):
    rc = cli.main(["reading", "--date", "2024-04-08T18:17:00Z", "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["eclipse"]["is_eclipse"] is True
    assert out["positions"]["Sun"]["longitude_deg"] >= 0.0
    # Mercury was retrograde during the April 2024 eclipse.
    assert out["positions"]["Mercury"]["retrograde"] is True


def test_reading_with_location(capsys):
    rc = cli.main(["reading", "--date", "2026-08-22T12:00",
                   "--lat", "51.5", "--lon", "-0.12", "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert "location" in out and out["location"]["local_intensity"] > 0.0


def test_map_writes_real_field(tmp_path, capsys):
    out = tmp_path / "hm.json"
    rc = cli.main(["map", "--date", "2024-04-08T18:17:00Z",
                   "--nodes", "300", "--out", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert len(payload["potential"]) == 300
    assert payload["summary"]["eclipse"] is True
    assert max(payload["potential"]) > min(payload["potential"])  # real variation


def test_scan_runs(capsys):
    rc = cli.main(["scan", "--start", "2024-09-01", "--end", "2024-10-05",
                   "--step-hours", "24", "--top", "3"])
    assert rc == 0
    assert "ECLIPSE" in capsys.readouterr().out  # Sept/Oct 2024 eclipses
