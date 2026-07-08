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


def parse_gpu_sample(nvidia_smi_out: str) -> dict:
    """Parse one CSV line from
    `nvidia-smi --query-gpu=utilization.gpu,power.draw,memory.used,memory.total
     --format=csv,noheader,nounits` into a dict. Pure, unit-testable."""
    line = nvidia_smi_out.strip().splitlines()[0]
    parts = [p.strip() for p in line.split(",")]
    return {
        "gpu_util_pct": float(parts[0]),
        "gpu_power_w": float(parts[1]),
        "gpu_mem_used_mb": float(parts[2]),
        "gpu_mem_total_mb": float(parts[3]),
    }


def query_gpu_sample() -> dict:
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,power.draw,memory.used,memory.total",
             "--format=csv,noheader,nounits"], text=True)
        return parse_gpu_sample(out)
    except Exception:
        return {"gpu_util_pct": -1.0, "gpu_power_w": -1.0,
                "gpu_mem_used_mb": -1.0, "gpu_mem_total_mb": -1.0}


def parse_loadavg(loadavg_text: str) -> dict:
    """Parse /proc/loadavg first three fields. Pure."""
    f = loadavg_text.split()
    return {"load1": float(f[0]), "load5": float(f[1]), "load15": float(f[2])}


def cpu_load() -> dict:
    try:
        return parse_loadavg(Path("/proc/loadavg").read_text())
    except Exception:
        return {"load1": -1.0, "load5": -1.0, "load15": -1.0}


def host_sample() -> dict:
    """One combined GPU+CPU+RAM resource sample (no timestamp; caller stamps)."""
    s = query_gpu_sample()
    s.update(cpu_load())
    s["ram_avail_gb"] = free_ram_gb()
    return s


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
