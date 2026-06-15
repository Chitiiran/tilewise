"""Daily-runner config knobs (spec 2026-06-13 §12)."""
from __future__ import annotations


def test_daily_knobs_defaults():
    from catan_az.config import AzConfig
    c = AzConfig()
    assert c.fresh_ratio == 0.70
    assert c.rules_id == "v3-full"
    assert c.worker_procs_max == 7
    assert c.per_proc_vram_mb == 535.0
    assert c.worker_nice == 10
    assert c.min_fast_gb == 10.0
    assert c.min_hdd_gb == 20.0
    assert c.stagnation_holds == 5
    assert c.archive_root == "/mnt/d/catan_az_archive"
    assert c.dashboard_port == 8099
