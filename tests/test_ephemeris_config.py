"""Config save/load + auto_configure. Restores the default backend after each test."""

import pytest

from kalachakra.ephemeris import global_state as gs


@pytest.fixture(autouse=True)
def _restore_backend():
    yield
    gs.configure(mode="moshier")   # never leave a bogus swiss path set


def test_save_config_writes_json(tmp_path):
    p = gs.save_config(mode="swiss", ephe_path="/data/ephe", path=tmp_path / "c.json")
    assert p.exists()
    import json
    data = json.loads(p.read_text())
    assert data == {"mode": "swiss", "ephe_path": "/data/ephe", "jpl_file": None}


def test_auto_configure_env_ephe_path(monkeypatch):
    monkeypatch.setenv("KALACHAKRA_EPHE_PATH", "/some/ephe")
    monkeypatch.delenv("KALACHAKRA_JPL_FILE", raising=False)
    assert gs.auto_configure() == "swiss"
    assert gs._MODE == "swiss"


def test_auto_configure_env_jpl_wins(monkeypatch):
    monkeypatch.setenv("KALACHAKRA_JPL_FILE", "/some/de441.bsp")
    assert gs.auto_configure() == "jpl"


def test_auto_configure_reads_config_file(tmp_path, monkeypatch):
    cfg = tmp_path / "c.json"
    gs.save_config(mode="swiss", ephe_path="/x/ephe", path=cfg)
    monkeypatch.delenv("KALACHAKRA_EPHE_PATH", raising=False)
    monkeypatch.delenv("KALACHAKRA_JPL_FILE", raising=False)
    monkeypatch.setenv("KALACHAKRA_CONFIG", str(cfg))
    assert gs.auto_configure() == "swiss"


def test_auto_configure_defaults_to_moshier(tmp_path, monkeypatch):
    monkeypatch.delenv("KALACHAKRA_EPHE_PATH", raising=False)
    monkeypatch.delenv("KALACHAKRA_JPL_FILE", raising=False)
    # No config files anywhere.
    monkeypatch.setattr(gs, "_config_search_paths", lambda: [tmp_path / "nope.json"])
    assert gs.auto_configure() == "moshier"


def test_configure_from_args_explicit_wins(monkeypatch):
    monkeypatch.setenv("KALACHAKRA_JPL_FILE", "/ignored.bsp")
    # Explicit ephe_path must beat the env var.
    assert gs.configure_from_args(ephe_path="/explicit") == "swiss"
