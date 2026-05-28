"""Background memory observer. Samples /proc/meminfo + a target PID's
/proc/<pid>/status every 5s, writes JSON-per-line to --log.

Used alongside training: launch this in the background pinned to the
trainer's PID. If the trainer dies (host-OOM, WSL shutdown, etc), the
observer's last few samples tell us the trajectory.

Exits cleanly when the target PID is gone.

Usage:
    python observe_memory.py --pid <PID> --log <path> [--interval 5]
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path


def _read_meminfo() -> dict:
    out = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":", 1)
            out[k] = int(v.strip().split()[0])
    return out


def _read_pid_status(pid: int) -> dict | None:
    try:
        out = {}
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if ":" not in line:
                    continue
                k, v = line.split(":", 1)
                out[k.strip()] = v.strip()
        return out
    except (FileNotFoundError, ProcessLookupError):
        return None


def _emit(log_path: Path, label: str, pid: int, **extra) -> bool:
    """Write one sample. Returns False if PID is dead (caller exits)."""
    st = _read_pid_status(pid)
    if st is None:
        rec = {"ts": time.time(), "label": "pid_gone", "pid": pid, **extra}
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        return False

    mem = _read_meminfo()
    rss = int(st.get("VmRSS", "0").split()[0])
    swap = int(st.get("VmSwap", "0").split()[0])
    rec = {
        "ts": time.time(),
        "label": label,
        "pid": pid,
        "rss_gb": rss / 1024 / 1024,
        "swap_gb": swap / 1024 / 1024,
        "sys_free_gb": mem["MemFree"] / 1024 / 1024,
        "sys_available_gb": mem["MemAvailable"] / 1024 / 1024,
        "sys_buff_cache_gb": (mem.get("Buffers", 0) + mem.get("Cached", 0)) / 1024 / 1024,
        "sys_swap_used_gb": (mem["SwapTotal"] - mem["SwapFree"]) / 1024 / 1024,
        **extra,
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return True


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pid", type=int, required=True, help="PID to observe")
    p.add_argument("--log", type=Path, required=True, help="Output log path (JSON per line)")
    p.add_argument("--interval", type=float, default=5.0, help="Sample period in seconds")
    args = p.parse_args()

    args.log.parent.mkdir(parents=True, exist_ok=True)
    # Touch the log (don't truncate — append-mode, so multiple runs accumulate)
    args.log.touch()

    if not _emit(args.log, "startup", args.pid):
        print(f"[observer] target pid {args.pid} already gone; exiting", flush=True)
        return 0

    while True:
        time.sleep(args.interval)
        alive = _emit(args.log, "tick", args.pid)
        if not alive:
            print(f"[observer] target pid {args.pid} gone; exiting", flush=True)
            return 0


if __name__ == "__main__":
    sys.exit(main())
