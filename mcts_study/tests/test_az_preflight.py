"""Preflight guards (spec 2026-06-13 §6) — probes monkeypatched, no real GPU."""
from __future__ import annotations


def _patch(monkeypatch, **kw):
    import catan_az.resources as r
    monkeypatch.setattr(r, "query_free_vram_mb", lambda: kw.get("vram", 4096))
    monkeypatch.setattr(r, "free_disk_gb", lambda p: kw.get("disk", 100))
    monkeypatch.setattr(r, "free_ram_gb", lambda: kw.get("ram", 50))
    monkeypatch.setattr(r, "foreign_selfplay_pids",
                        lambda pid: kw.get("foreign", []))


def test_preflight_ok_caps_procs(monkeypatch, tmp_path):
    from catan_az.preflight import preflight
    from catan_az.config import AzConfig
    _patch(monkeypatch, vram=4096)
    res = preflight(AzConfig(), loop_root=tmp_path, archive_root=tmp_path)
    assert res.ok
    assert res.capped_procs == 7


def test_preflight_low_disk_aborts(monkeypatch, tmp_path):
    from catan_az.preflight import preflight
    from catan_az.config import AzConfig
    _patch(monkeypatch, disk=2)
    res = preflight(AzConfig(), loop_root=tmp_path, archive_root=tmp_path)
    assert not res.ok
    assert any("disk" in r.lower() for r in res.reasons)


def test_preflight_low_vram_degrades_not_abort(monkeypatch, tmp_path):
    from catan_az.preflight import preflight
    from catan_az.config import AzConfig
    _patch(monkeypatch, vram=1600)
    res = preflight(AzConfig(), loop_root=tmp_path, archive_root=tmp_path)
    assert res.ok and res.capped_procs == 2


def test_preflight_no_vram_aborts(monkeypatch, tmp_path):
    from catan_az.preflight import preflight
    from catan_az.config import AzConfig
    _patch(monkeypatch, vram=300)
    res = preflight(AzConfig(), loop_root=tmp_path, archive_root=tmp_path)
    assert not res.ok
