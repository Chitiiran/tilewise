"""Run guards before each daily cycle. Hard failures -> ok=False (abort with
reasons). Soft -> degrade (capped_procs). VRAM is the binding limit on the
4GB card; disk/lock are hard. GPU-busy is intentionally NOT a guard (user
shares the GPU; training is CPU-bound). Spec 2026-06-13 §6."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from . import resources as r


@dataclass
class PreflightResult:
    ok: bool
    capped_procs: int = 0
    reasons: list[str] = field(default_factory=list)


def preflight(cfg, *, loop_root: Path, archive_root: Path,
              my_pid: int | None = None) -> PreflightResult:
    reasons: list[str] = []
    my_pid = my_pid if my_pid is not None else os.getpid()

    # Hard: fast disk free
    fast = r.free_disk_gb(loop_root)
    if fast < cfg.min_fast_gb:
        reasons.append(f"fast-disk free {fast:.1f}GB < {cfg.min_fast_gb}GB")

    # Hard: HDD (archive target) free + mounted
    try:
        hdd = r.free_disk_gb(archive_root)
        if hdd < cfg.min_hdd_gb:
            reasons.append(f"hdd free {hdd:.1f}GB < {cfg.min_hdd_gb}GB")
    except FileNotFoundError:
        reasons.append(f"archive_root {archive_root} not mounted")

    # Soft: VRAM -> proc cap (0 fit is a hard abort)
    vram = r.query_free_vram_mb()
    procs = r.fit_gpu_procs(free_vram_mb=vram, per_proc_mb=cfg.per_proc_vram_mb,
                            hard_max=cfg.worker_procs_max)
    if procs == 0:
        reasons.append(f"VRAM free {vram:.0f}MB fits 0 procs")

    # Foreign procs are reaped by the caller (daily.py / entry script), not
    # here — preflight stays side-effect-free for testability.

    ok = len(reasons) == 0
    return PreflightResult(ok=ok, capped_procs=procs, reasons=reasons)
