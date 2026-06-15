"""Resource probes for the daily runner's guards (spec 2026-06-13 §6)."""
from __future__ import annotations


def test_fit_gpu_procs_caps_by_vram():
    from catan_az.resources import fit_gpu_procs
    assert fit_gpu_procs(free_vram_mb=4096, per_proc_mb=535, hard_max=7) == 7
    assert fit_gpu_procs(free_vram_mb=2000, per_proc_mb=535, hard_max=7) == 3
    assert fit_gpu_procs(free_vram_mb=400, per_proc_mb=535, hard_max=7) == 0


def test_parse_nvidia_smi_free_vram():
    from catan_az.resources import parse_free_vram_mb
    assert parse_free_vram_mb("3500\n") == 3500.0


def test_parse_disk_free_gb(tmp_path):
    from catan_az.resources import free_disk_gb
    assert free_disk_gb(tmp_path) > 0
