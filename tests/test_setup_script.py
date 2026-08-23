"""Tests for scripts/setup_full_span.py plumbing (no network needed)."""

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "setup_full_span.py"
_spec = importlib.util.spec_from_file_location("setup_full_span", _SCRIPT)
setup = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(setup)


def test_build_se1_plan_default_span(tmp_path):
    plan = setup.build_se1_plan(tmp_path, setup.DEFAULT_BASE_URL,
                                start_year=-3101, end_year=7154, mirror_all=False)
    assert len(plan) == 36
    urls = [u for u, _ in plan]
    assert any(u.endswith("/seplm36.se1") for u in urls)
    assert any(u.endswith("/sepl_66.se1") for u in urls)
    assert any(u.endswith("/semo_66.se1") for u in urls)
    # local paths land in dest
    assert all(p.parent == tmp_path for _, p in plan)


def test_build_jpl_plan(tmp_path):
    plan = setup.build_jpl_plan(tmp_path)
    assert [p.name for _, p in plan] == ["de441_part-1.bsp", "de441_part-2.bsp"]


def test_download_file_from_local_uri(tmp_path):
    src = tmp_path / "src.se1"
    src.write_bytes(b"SWISS-EPHEMERIS-FAKE" + b"\x00" * 4096)
    dest = tmp_path / "out" / "sepl_18.se1"
    ok = setup.download_file(src.as_uri(), dest)
    assert ok and dest.exists()
    assert dest.read_bytes() == src.read_bytes()


def test_download_file_skips_existing(tmp_path):
    dest = tmp_path / "sepl_18.se1"
    dest.write_bytes(b"x" * 5000)
    # A bogus URL would fail if it tried to fetch; existing file -> skip -> True.
    assert setup.download_file("http://invalid.invalid/nope.se1", dest)


def test_download_file_rejects_html(tmp_path):
    src = tmp_path / "err.html"
    src.write_bytes(b"<html>404 Not Found</html>")
    dest = tmp_path / "out.se1"
    ok = setup.download_file(src.as_uri(), dest, retries=1)
    assert not ok and not dest.exists()


def test_default_source_is_github_mirror():
    assert "raw.githubusercontent.com/aloistr/swisseph" in setup.DEFAULT_BASE_URL


def test_dry_run_lists_without_downloading(tmp_path, capsys):
    rc = setup.main(["--dry-run", "--dest", str(tmp_path / "ephe")])
    assert rc == 0
    out = capsys.readouterr().out
    assert "seplm36.se1" in out and "sepl_66.se1" in out
    assert not (tmp_path / "ephe").exists()  # nothing downloaded


def test_configure_only_with_existing_files(tmp_path, capsys):
    from kalachakra.ephemeris import se1_files
    dest = tmp_path / "ephe"
    dest.mkdir()
    for name in se1_files.filenames_for_years():
        (dest / name).write_bytes(b"FAKE" + b"\x00" * 4096)
    rc = setup.main(["--configure-only", "--dest", str(dest),
                     "--config-path", str(tmp_path / ".kalachakra.json"),
                     "--no-verify"])
    assert rc == 0
    assert (tmp_path / ".kalachakra.json").exists()


def test_configure_only_reports_missing(tmp_path, capsys):
    dest = tmp_path / "ephe"
    dest.mkdir()  # empty -> everything missing
    rc = setup.main(["--configure-only", "--dest", str(dest), "--no-verify"])
    assert rc == 1
    assert "missing" in capsys.readouterr().err.lower()
