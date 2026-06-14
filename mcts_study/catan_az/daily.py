"""Resumable daily AZ driver. run_day() runs cycles until a STOP sentinel or
max_iters. Each cycle is fresh-ratio self-play + loop.run_iteration. A manifest
(daily_state.json) makes a kill resumable to the exact stage; per-game flushes
(in the engine) bound loss to <=1 game. Spec 2026-06-13 §4-5."""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from .buffer import fresh_deficit


@dataclass
class DailyManifest:
    iter: int
    stage: str
    champion: str
    fresh_target: int
    fresh_done: int
    rules_id: str

    def save(self, loop_root: Path) -> None:
        p = Path(loop_root) / "daily_state.json"
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2))
        os.replace(tmp, p)

    @classmethod
    def load(cls, loop_root: Path) -> "DailyManifest | None":
        p = Path(loop_root) / "daily_state.json"
        return cls(**json.loads(p.read_text())) if p.exists() else None


def _stop_requested(loop_root: Path) -> bool:
    return (Path(loop_root) / "STOP").exists()


def _next_iter_number(loop_root: Path) -> int:
    nums = []
    for d in Path(loop_root).glob("iter_*"):
        part = d.name.split("_", 1)[1]
        if part.isdigit():
            nums.append(int(part))
    return (max(nums) + 1) if nums else 1


def run_day(cfg, *, loop_root: Path, capped_procs: int, cycle_fn,
            max_iters: int = 1000, next_iter: int | None = None) -> None:
    """Run cycles until STOP / max_iters. cycle_fn(cfg, loop_root, iter_n,
    capped_procs) -> verdict."""
    loop_root = Path(loop_root)
    loop_root.mkdir(parents=True, exist_ok=True)
    n = next_iter if next_iter is not None else _next_iter_number(loop_root)
    done = 0
    while done < max_iters:
        if _stop_requested(loop_root):
            break
        cycle_fn(cfg, loop_root, n, capped_procs)
        n += 1
        done += 1


def _launch_selfplay_procs(cfg, out_dir, checkpoint, n_games, n_procs,
                           champion, rules_id):
    """Launch n_procs LOW-PRIORITY (nice) self_play_async procs splitting
    n_games, blocking until all exit. Each run dir is tagged with meta.json
    {rules_id, champion} (the experiment doesn't know about them). Returns the
    run dirs created. Per-game flush in the engine bounds loss to <=1 game on
    a kill. (Spec §3b: nice workers yield to foreground work.)"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per = max(1, n_games // max(1, n_procs))
    procs = []
    for i in range(n_procs):
        sb = 31_000_000 + i * 1_000_000
        cmd = ["nice", "-n", str(cfg.worker_nice),
               "python", "-m", "catan_mcts.experiments.self_play_async",
               "--out-root", str(out_dir), "--checkpoint", str(checkpoint),
               "--num-games", str(per), "--n-sims", str(cfg.sims),
               "--n-concurrent", str(cfg.n_concurrent),
               "--max-batch", str(cfg.max_batch), "--self-play",
               "--seed-base", str(sb), "--device", "cuda",
               "--max-seconds", "21600"]
        procs.append(subprocess.Popen(cmd))
    for p in procs:
        p.wait()
    dirs = sorted(out_dir.glob("*self_play_async*"))
    for d in dirs:
        (d / "meta.json").write_text(
            json.dumps({"rules_id": rules_id, "champion": champion}))
    return dirs


def generate_fresh(cfg, *, iter_dir: Path, champion: str, champion_ckpt: Path,
                   capped_procs: int, prior_dirs: list) -> list:
    """Generate current-champion self-play until fresh games >= fresh_ratio of
    the window. Resumable: counts existing fresh first, generates only the
    deficit (so a kill mid-self-play resumes toward the target, never
    regenerates). Spec §5."""
    deficit = fresh_deficit(prior_dirs, champion=champion, rules_id=cfg.rules_id,
                            window_games=cfg.window_games,
                            fresh_ratio=cfg.fresh_ratio)
    if deficit <= 0:
        return []
    return _launch_selfplay_procs(cfg, Path(iter_dir) / "selfplay",
                                  champion_ckpt, deficit, capped_procs,
                                  champion, cfg.rules_id)
