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
from .buffer import fresh_deficit, gen_window, select_window
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
        # Terminal: max_iters_per_model iterations since the last promotion ->
        # STOP and surface for the user to decide next (spec 2026-06-14 §4 step 7).
        holds = _holds_since_promotion(loop_root / "journal.csv")
        if holds >= cfg.max_iters_per_model:
            StatusWriter(loop_root).stage(n - 1, "max_iters_reached",
                                          MAX_ITERS=True, holds_since_promotion=holds)
            break


def _launch_selfplay_procs(cfg, out_dir, checkpoint, n_games, n_procs,
                           generator_name, gen_iter, rules_id):
    """Launch n_procs LOW-PRIORITY (nice) self_play_async procs splitting
    n_games, blocking until all exit. Each run dir is tagged with meta.json
    {rules_id, generator_name, gen_iter} — recording WHICH net made the games
    and WHICH iteration, the dimension whose absence caused the stale-data bug
    (2026-06-14). Per-game flush bounds loss to <=1 game on a kill."""
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
        (d / "meta.json").write_text(json.dumps(
            {"rules_id": rules_id, "generator_name": generator_name,
             "gen_iter": gen_iter}))
    return dirs


def generate_iter_games(cfg, *, iter_dir: Path, generator, gen_iter: int,
                        capped_procs: int, prior_dirs: list) -> list:
    """Generate exactly cfg.games_per_iter NEW games with `generator` for this
    iteration (spec 2026-06-14 §4). Resumable: counts ONLY this iteration's own
    gen_iter games and generates the remainder — so a kill resumes toward the
    quota, and other iterations' games never satisfy it (the stale-data bug)."""
    from .buffer import count_games, own_iter_games
    gen_name, gen_ckpt = generator
    have = own_iter_games(prior_dirs, gen_iter=gen_iter)
    deficit = max(0, cfg.games_per_iter - have)
    if deficit <= 0:
        return []
    dirs = _launch_selfplay_procs(cfg, Path(iter_dir) / "selfplay",
                                  Path(gen_ckpt), deficit, capped_procs,
                                  gen_name, gen_iter, cfg.rules_id)
    produced = sum(count_games(d) for d in dirs)
    if produced == 0:
        raise RuntimeError(
            f"self-play produced 0 games for iter {gen_iter} (generator "
            f"{gen_name}, ckpt {gen_ckpt}) — check proc logs (arch / OOM).")
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
    """One full AZ cycle (redesign 2026-06-14): self-play with the LATEST
    candidate -> gen_iter window (reset on promotion) -> train/arena/publish ->
    archive. Returns the verdict. Spec 2026-06-14 §4."""
    loop_root = Path(loop_root)
    iter_dir = loop_root / f"iter_{iter_n}"
    ladder = Ladder(loop_root)
    champion = (ladder.champion()["name"], ladder.champion()["checkpoint"])
    boundary = ladder.last_promotion_iter()      # window reset boundary

    generator = select_generator(loop_root, iter_n, champion)
    DailyManifest(iter=iter_n, stage="selfplay", champion=champion[0],
                  fresh_target=cfg.games_per_iter, fresh_done=0,
                  rules_id=cfg.rules_id).save(loop_root)

    prior = _all_selfplay_dirs(loop_root)
    new_dirs = generate_iter_games(cfg, iter_dir=iter_dir, generator=generator,
                                   gen_iter=iter_n, capped_procs=capped_procs,
                                   prior_dirs=prior)

    all_dirs = new_dirs + prior
    window = gen_window(all_dirs, gen_iter_min=boundary, rules_id=cfg.rules_id)

    DailyManifest(iter=iter_n, stage="iterate", champion=champion[0],
                  fresh_target=cfg.games_per_iter, fresh_done=len(new_dirs),
                  rules_id=cfg.rules_id).save(loop_root)

    verdict = run_iteration(cfg, loop_root, iter_n,
                            existing_selfplay_dirs=[str(d) for d in window])

    # On promote, run_iteration's PUBLISH already crowned the candidate; set the
    # window-reset boundary (idempotent — no duplicate history entry).
    if verdict == "promote":
        Ladder(loop_root).promote(f"az_iter_{iter_n}", promoted_at_iter=iter_n)

    archive_out_of_window(window_dirs=window, all_dirs=all_dirs,
                          archive_root=Path(cfg.archive_root),
                          rules_id=cfg.rules_id)

    _append_progress_row(loop_root, iter_n, generator[0], new_dirs, window)
    DailyManifest(iter=iter_n, stage="done", champion=champion[0],
                  fresh_target=cfg.games_per_iter, fresh_done=len(new_dirs),
                  rules_id=cfg.rules_id).save(loop_root)
    return verdict


def _holds_since_promotion(journal_path: Path) -> int:
    """Iterations since the last promotion (hold/invalid verdicts). The
    'max_iters_per_model' terminal counts THIS — 'how many iters has the current
    champion gone unbeaten' — not raw trailing holds (spec 2026-06-14 §4 step 7)."""
    if not Path(journal_path).exists():
        return 0
    with open(journal_path, newline="") as f:
        rows = list(_csv.DictReader(f))
    n = 0
    for row in reversed(rows):
        if row.get("verdict") == "promote":
            break
        n += 1
    return n


def _iters_of_dirs(dirs) -> list:
    """Which iter_<N> each run dir belongs to (by path), de-duped."""
    out = set()
    for d in dirs:
        for part in Path(d).parts:
            if part.startswith("iter_") and part.split("_", 1)[1].isdigit():
                out.add(int(part.split("_", 1)[1]))
    return sorted(out)


def _window_iters(window) -> list:
    """The gen_iters spanned by a window (from meta.json), de-duped + sorted."""
    from .buffer import _read_meta
    out = {int(_read_meta(d).get("gen_iter", 0)) for d in window}
    return sorted(out)


def _append_progress_row(loop_root, iter_n, generator, new_dirs, window):
    from .buffer import count_games
    from .progress import append_progress
    import csv as _c
    new_games = sum(count_games(d) for d in new_dirs)
    champion = Ladder(Path(loop_root)).champion()["name"]
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
                    generator=generator, new_games=new_games,
                    window_dirs=len(window), window_iters=_window_iters(window),
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
