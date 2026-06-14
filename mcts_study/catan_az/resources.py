"""Pure resource probes for the daily runner's guards. No side effects
beyond reading system state; each function is unit-testable with a string
or a path so preflight can be tested without a GPU. Spec 2026-06-13 §6."""
from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path


def fit_gpu_procs(*, free_vram_mb: float, per_proc_mb: float, hard_max: int) -> int:
    """How many GPU self-play procs fit in free VRAM, capped at hard_max."""
    by_vram = int(math.floor(free_vram_mb / per_proc_mb))
    return max(0, min(hard_max, by_vram))


def parse_free_vram_mb(nvidia_smi_out: str) -> float:
    """Parse `nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits`."""
    line = nvidia_smi_out.strip().splitlines()[0]
    return float(line.strip())


def query_free_vram_mb() -> float:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free",
         "--format=csv,noheader,nounits"], text=True)
    return parse_free_vram_mb(out)


def free_disk_gb(path) -> float:
    usage = shutil.disk_usage(Path(path))
    return usage.free / (1024 ** 3)


def free_ram_gb() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / (1024 ** 2)
    return 0.0


def foreign_selfplay_pids(my_pid: int) -> list[int]:
    """PIDs of stray self_play_async / catan_az.daily / .loop procs not me."""
    try:
        out = subprocess.check_output(
            ["pgrep", "-f",
             "experiments.self_play_async|catan_az.daily|catan_az.loop"],
            text=True)
    except subprocess.CalledProcessError:
        return []
    return [int(p) for p in out.split() if int(p) != my_pid]
