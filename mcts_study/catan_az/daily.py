"""Resumable daily AZ driver. run_day() runs cycles until a STOP sentinel or
max_iters. Each cycle is fresh-ratio self-play + loop.run_iteration. A manifest
(daily_state.json) makes a kill resumable to the exact stage; per-game flushes
(in the engine) bound loss to <=1 game. Spec 2026-06-13 §4-5."""
from __future__ import annotations

import csv as _csv
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from .archive import archive_out_of_window
from .buffer import fresh_deficit, select_window
from .ladder import Ladder
from .loop import run_iteration
from .status import StatusWriter


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
    """The iteration to run next. If the latest iteration started but never
    PUBLISHed (crash/interrupt mid-cycle), RESUME it — its self-play + train
    are salvaged via done-markers — instead of abandoning that work and
    starting a fresh number (2026-06-14: a training crash wasted 6h of iter-3
    self-play when resume jumped to iter-4)."""
    nums = []
    for d in Path(loop_root).glob("iter_*"):
        part = d.name.split("_", 1)[1]
        if part.isdigit():
            nums.append(int(part))
    if not nums:
        return 1
    latest = max(nums)
    # Incomplete latest iteration (dir exists, no PUBLISH.done) -> resume it.
    if not (Path(loop_root) / f"iter_{latest}" / "PUBLISH.done").exists():
        return latest
    return latest + 1


def stagnation_holds_from_journal(journal_path: Path) -> int:
    """Trailing consecutive non-promote verdicts (hold/invalid) — surfaced as
    a flag, NOT an error (a HOLD is a result). Spec §7."""
    if not Path(journal_path).exists():
        return 0
    with open(journal_path, newline="") as f:
        rows = list(_csv.DictReader(f))
    n = 0
    for row in reversed(rows):
        if row.get("verdict") in ("hold", "invalid"):
            n += 1
        else:
            break
    return n


def run_day(cfg, *, loop_root: Path, capped_procs: int, cycle_fn,
            max_iters: int = 1000, next_iter: int | None = None) -> None:
    """Run cycles until STOP / max_iters / stagnation. cycle_fn(cfg, loop_root,
    iter_n, capped_procs) -> verdict."""
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
        # Surface stagnation (not an error) and stop the day so compute isn't
        # burned grinding a stuck champion. Resumes next run.
        holds = stagnation_holds_from_journal(loop_root / "journal.csv")
        if holds >= cfg.stagnation_holds:
            StatusWriter(loop_root).stage(n - 1, "stagnation",
                                          STAGNATION=True, trailing_holds=holds)
            break


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
               # arch must match the champion ckpt, else load size-mismatch
               "--hidden-dim", str(cfg.hidden_dim),
               "--num-layers", str(cfg.num_layers),
               "--vp-target", str(cfg.vp_target),
               "--max-seconds", "21600"]
        if not cfg.bonuses:
            cmd.append("--no-bonuses")
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
    dirs = _launch_selfplay_procs(cfg, Path(iter_dir) / "selfplay",
                                  champion_ckpt, deficit, capped_procs,
                                  champion, cfg.rules_id)
    # Resilience: a crashed self-play proc leaves an empty dir; that would
    # surface as a cryptic "no games in window" later. Fail loud + located
    # here instead (e.g. arch mismatch, OOM). Spec §2 (env failures surfaced).
    from .buffer import count_games
    produced = sum(count_games(d) for d in dirs)
    if produced == 0:
        raise RuntimeError(
            f"self-play produced 0 games in {len(dirs)} dir(s) under "
            f"{Path(iter_dir) / 'selfplay'} — check the proc logs (arch "
            f"mismatch / OOM / bad checkpoint {champion_ckpt}).")
    return dirs


def select_generator(loop_root, iter_n: int, champion):
    """Net that generates THIS iteration's self-play (spec 2026-06-14 §4).
    iter-1, or no prior candidate -> champion; else iter (N-1)'s trained
    candidate (promoted or not — self-play always uses the LATEST net, the
    canonical-AZ separation of 'who self-plays' from 'who's crowned').
    Returns (generator_name, ckpt_path)."""
    prev = Path(loop_root) / f"iter_{iter_n - 1}" / "training" / "checkpoint_best.pt"
    if iter_n > 1 and prev.exists():
        return f"cand_iter_{iter_n - 1}", str(prev)
    return champion   # (name, ckpt)


def _champion_from_ladder(loop_root: Path):
    c = Ladder(Path(loop_root)).champion()
    return c["name"], c["checkpoint"]


def _all_selfplay_dirs(loop_root: Path) -> list:
    """All prior self-play run dirs, newest-iteration first."""
    dirs = []
    iters = sorted(Path(loop_root).glob("iter_*"),
                   key=lambda p: int(p.name.split("_", 1)[1])
                   if p.name.split("_", 1)[1].isdigit() else -1, reverse=True)
    for it in iters:
        sp = it / "selfplay"
        if sp.exists():
            dirs.extend(sorted(sp.glob("*self_play_async*"), reverse=True))
    return dirs


def run_cycle(cfg, loop_root: Path, iter_n: int, capped_procs: int) -> str:
    """One full AZ cycle: fresh self-play -> run_iteration (train/arena/publish)
    -> archive out-of-window games to HDD. Writes the manifest at each
    transition. Returns the verdict. Spec §4-5, §8."""
    loop_root = Path(loop_root)
    iter_dir = loop_root / f"iter_{iter_n}"
    champion, champion_ckpt = _champion_from_ladder(loop_root)

    DailyManifest(iter=iter_n, stage="selfplay", champion=champion,
                  fresh_target=0, fresh_done=0,
                  rules_id=cfg.rules_id).save(loop_root)

    prior = _all_selfplay_dirs(loop_root)
    fresh_dirs = generate_fresh(cfg, iter_dir=iter_dir, champion=champion,
                                champion_ckpt=Path(champion_ckpt),
                                capped_procs=capped_procs, prior_dirs=prior)

    DailyManifest(iter=iter_n, stage="iterate", champion=champion,
                  fresh_target=0, fresh_done=len(fresh_dirs),
                  rules_id=cfg.rules_id).save(loop_root)

    all_fresh = fresh_dirs + prior
    verdict = run_iteration(cfg, loop_root, iter_n,
                            existing_selfplay_dirs=[str(d) for d in all_fresh])

    # Archive out-of-window games to HDD (after publish, never deletes).
    try:
        window = select_window(all_fresh, cfg.window_games, rules_id=cfg.rules_id)
    except ValueError:
        window = all_fresh
    archive_out_of_window(window_dirs=window, all_dirs=all_fresh,
                          archive_root=Path(cfg.archive_root),
                          rules_id=cfg.rules_id)

    DailyManifest(iter=iter_n, stage="done", champion=champion,
                  fresh_target=0, fresh_done=len(fresh_dirs),
                  rules_id=cfg.rules_id).save(loop_root)

    # PROGRESS.md: the at-a-glance "what did this iter train on + did it make
    # new data?" record (the question that needed archaeology on 2026-06-14).
    _append_progress_row(loop_root, iter_n, champion, fresh_dirs, window)
    return verdict


def _iters_of_dirs(dirs) -> list:
    """Which iter_<N> each run dir belongs to (by path), de-duped."""
    out = set()
    for d in dirs:
        for part in Path(d).parts:
            if part.startswith("iter_") and part.split("_", 1)[1].isdigit():
                out.add(int(part.split("_", 1)[1]))
    return sorted(out)


def _append_progress_row(loop_root, iter_n, champion, fresh_dirs, window):
    from .buffer import count_games
    from .progress import append_progress
    import csv as _c
    new_games = sum(count_games(d) for d in fresh_dirs)
    # read the just-published journal row for winrate/draws
    wr = dr = 0.0
    verdict = "?"
    jp = Path(loop_root) / "journal.csv"
    if jp.exists():
        rows = list(_c.DictReader(open(jp)))
        for r in reversed(rows):
            if str(r.get("iter")) == str(iter_n):
                wr = float(r.get("arena_winrate", 0) or 0)
                c = int(r.get("arena_wins_cand", 0) or 0)
                ch = int(r.get("arena_wins_champ", 0) or 0)
                d = int(r.get("arena_draws", 0) or 0)
                g = c + ch + d
                dr = d / g if g else 0.0
                verdict = r.get("verdict", "?")
                break
    append_progress(loop_root, iter_n=iter_n, champion=champion,
                    new_games=new_games, window_games=len(window),
                    window_dirs=len(window), all_from_iters=_iters_of_dirs(window),
                    verdict=verdict, winrate=wr, draw_rate=dr)


def cli_main():
    """Entry: preflight -> run_day. Used by scripts/run_az_day.sh."""
    import argparse

    from .config import AzConfig
    from .preflight import preflight
    p = argparse.ArgumentParser()
    p.add_argument("--loop-root", type=Path, required=True)
    p.add_argument("--max-iters", type=int, default=1000)
    p.add_argument("--config", type=Path, default=None)
    args = p.parse_args()
    cfg = AzConfig.from_json(args.config) if args.config else AzConfig()
    res = preflight(cfg, loop_root=args.loop_root,
                    archive_root=Path(cfg.archive_root))
    if not res.ok:
        print("[az-day] preflight FAILED:", "; ".join(res.reasons))
        raise SystemExit(1)
    print(f"[az-day] preflight ok, {res.capped_procs} procs")
    run_day(cfg, loop_root=args.loop_root, capped_procs=res.capped_procs,
            cycle_fn=run_cycle, max_iters=args.max_iters)


if __name__ == "__main__":
    cli_main()
