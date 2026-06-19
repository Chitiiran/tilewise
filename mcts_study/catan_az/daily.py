"""Resumable daily AZ driver. run_day() runs cycles until a STOP sentinel or
max_iters. Each cycle is fresh-ratio self-play + loop.run_iteration. A manifest
(daily_state.json) makes a kill resumable to the exact stage; per-game flushes
(in the engine) bound loss to <=1 game. Spec 2026-06-13 §4-5."""
from __future__ import annotations

import csv as _csv
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from .archive import archive_out_of_window
from .buffer import fresh_deficit, gen_window, select_window
from .ladder import Ladder
from .arena import ArenaPaused
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


def _pause_requested(loop_root: Path) -> bool:
    return (Path(loop_root) / "PAUSE").exists()


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
        # PAUSE at the top stops the loop cleanly (M1 fix, code review
        # 2026-06-19): the per-stage pausers also bail on PAUSE, but checking
        # here means the operator must REMOVE the PAUSE file before relaunch —
        # otherwise resume would re-enter and immediately re-pause on the first
        # chunk. Surfaced so it's not a silent no-op.
        if _pause_requested(loop_root):
            print("[az-day] PAUSE present — stopping. Remove the PAUSE file to resume.",
                  flush=True)
            StatusWriter(loop_root).stage(n, "paused", PAUSED=True)
            break
        try:
            cycle_fn(cfg, loop_root, n, capped_procs)
        except ArenaPaused as e:
            # A mid-arena pause: stop WITHOUT a verdict (results.jsonl durable;
            # resume completes). Do not advance the iter counter.
            print(f"[az-day] {e} — stopping (no verdict published).", flush=True)
            StatusWriter(loop_root).stage(n, "paused", PAUSED=True)
            break
        n += 1
        done += 1
        # Terminal: max_iters_per_model iterations since the last promotion ->
        # STOP and surface for the user to decide next (spec 2026-06-14 §4 step 7).
        holds = _holds_since_promotion(loop_root / "journal.csv")
        if holds >= cfg.max_iters_per_model:
            StatusWriter(loop_root).stage(n - 1, "max_iters_reached",
                                          MAX_ITERS=True, holds_since_promotion=holds)
            break


def _worker_seed_base(*, gen_iter: int, i: int) -> int:
    """Seed-base for self-play worker `i` in iteration `gen_iter`.

    M1 (deep-inspect HIGH): the seed-base MUST include the iteration term, else
    every iteration replays the identical board set and the run can only
    memorise a fixed ~1000 boards — scientifically worthless. The per-iter block
    (10M) is wide enough that a worker's 1000-game seed range never collides
    with another iter/worker across a 10-iter x 7-worker grid."""
    return 31_000_000 + gen_iter * 10_000_000 + i * 1_000_000


def _rust_cuda_env() -> dict:
    """Env for a Rust-engine self-play worker so tch uses the GPU.

    The pip-wheel libtorch needs: LD_PRELOAD libtorch_cuda.so (else
    tch::Cuda::is_available()==false), nvidia/*/lib on LD_LIBRARY_PATH (nvrtc
    for TorchScript kernel fusion), and deterministic kernels
    (CUBLAS_WORKSPACE_CONFIG + the .ts was traced in det mode) for the replay
    contract. Built from the active torch install. See
    project_task10_batching_decision_2026_06_18.
    """
    import os
    import torch

    torch_dir = os.path.dirname(torch.__file__)
    site = os.path.dirname(torch_dir)
    nvidia = os.path.join(site, "nvidia")
    nv_libs = []
    if os.path.isdir(nvidia):
        for d in os.listdir(nvidia):
            lib = os.path.join(nvidia, d, "lib")
            if os.path.isdir(lib):
                nv_libs.append(lib)
    paths = [os.path.join(torch_dir, "lib"), *nv_libs]
    env = dict(os.environ)
    prev = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join(paths + ([prev] if prev else []))
    preload = os.path.join(torch_dir, "lib", "libtorch_cuda.so")
    # Only inject the preload if the .so actually exists — a missing path in
    # LD_PRELOAD aborts the interpreter at startup (M2). Caller checks for its
    # presence to decide whether the GPU env is usable.
    if os.path.exists(preload):
        prev_pre = env.get("LD_PRELOAD", "")
        env["LD_PRELOAD"] = preload + (f":{prev_pre}" if prev_pre else "")
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    return env


def _launch_selfplay_procs(cfg, out_dir, checkpoint, n_games, n_procs,
                           generator_name, gen_iter, rules_id, seed_offset=0):
    """Launch n_procs LOW-PRIORITY (nice) self_play_async procs splitting
    n_games, blocking until all exit. Each run dir is tagged with meta.json
    {rules_id, generator_name, gen_iter} — recording WHICH net made the games
    and WHICH iteration, the dimension whose absence caused the stale-data bug
    (2026-06-14). Per-game flush bounds loss to <=1 game on a kill.

    `seed_offset` shifts every worker's seed-base (2026-06-15 resume fix): a
    deficit-resume re-launches into NEW dirs whose per-dir done.txt can't see
    the prior run's done seeds, so without an offset the workers would REPLAY
    the already-produced boards (duplicates). Offsetting by the count of games
    already on disk pushes the deficit games onto fresh seeds. The offset stays
    well within each worker's 1M-wide block (deficit << 1M), so no cross-worker
    seed collision."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    is_rust = getattr(cfg, "engine", "python") == "rust"
    # THROUGHPUT (Phase 3, GPU-forward-bound finding): the Rust engine batches
    # ALL its games into one GPU forward, so the optimal model is ONE process
    # with high n_concurrent (one fat batch ~93% full), NOT N small processes
    # each fragmenting the GPU into a tiny batch (that was right only for the old
    # 1-core asyncio engine). So force n_procs=1 for rust; the in-process chunked
    # batcher (self_play_rust) handles concurrency + pausability + resume.
    if is_rust:
        n_procs = 1
    per = max(1, n_games // max(1, n_procs))
    procs = []
    for i in range(n_procs):
        # Resume dedup (C2 fix, code review 2026-06-19): the rust path is ONE
        # process with a DETERMINISTIC run dir + --resume-dir, so resume reuses
        # the dir and its done.txt skip-list does the dedup. No seed_offset for
        # rust (offset + done-skip stacked could lose/dup games). The async path
        # keeps its per-worker-dir + offset scheme unchanged.
        sb = _worker_seed_base(gen_iter=gen_iter, i=i) + (0 if is_rust else seed_offset)
        sp_module = ("catan_mcts.experiments.self_play_rust" if is_rust
                     else "catan_mcts.experiments.self_play_async")
        cmd = ["nice", "-n", str(cfg.worker_nice),
               "python", "-m", sp_module,
               "--out-root", str(out_dir), "--checkpoint", str(checkpoint),
               "--num-games", str(per), "--n-sims", str(cfg.sims),
               "--n-concurrent", str(cfg.n_concurrent),
               "--max-batch", str(cfg.max_batch), "--self-play",
               "--seed-base", str(sb), "--device", "cuda",
               # arch must match the champion ckpt, else load size-mismatch
               "--hidden-dim", str(cfg.hidden_dim),
               "--num-layers", str(cfg.num_layers),
               "--vp-target", str(cfg.vp_target),
               "--max-seconds", str(cfg.selfplay_worker_max_seconds),
               "--game-deadline-seconds",
               str(cfg.selfplay_game_deadline_seconds)]
        if not cfg.bonuses:
            cmd.append("--no-bonuses")
        if is_rust:
            # Deterministic run dir + resume-dir so a kill/pause resumes INTO the
            # same dir and skips done seeds (C2). PAUSE sentinel checked in the
            # iter dir / loop root between chunks.
            # Name includes "self_play_rust" so the *self_play_* globs (tagging +
            # _all_selfplay_dirs) find it. Fixed (not timestamped) so resume reuses.
            rust_dir = out_dir / "self_play_rust"
            rust_dir.mkdir(parents=True, exist_ok=True)
            cmd += ["--resume-dir", str(rust_dir),
                    "--pause-dir", str(out_dir.parent)]
        # Rust engine: the tch-linked extension only uses the GPU with the
        # pip-wheel CUDA env (LD_PRELOAD libtorch_cuda, nvidia/*/lib on
        # LD_LIBRARY_PATH for nvrtc, deterministic kernels). Without it the Rust
        # self-play silently runs on CPU. No-op for the Python engine path.
        env = None
        if getattr(cfg, "engine", "python") == "rust":
            env = _rust_cuda_env()
        # start_new_session so the workers form their own process group: lets a
        # cleanup handler kill the whole group and avoids a stray Ctrl-C
        # propagating from the parent's terminal (deep-inspect MED: orphan GPU
        # workers / parent-death leakage).
        procs.append(subprocess.Popen(cmd, start_new_session=True, env=env))
    # M4 (deep-inspect HIGH): a per-worker wait timeout so one hung worker can't
    # stall the iteration for hours; kill + log if it overruns its budget. Must
    # exceed the worker's own wall-clock cap (else we'd kill a healthy worker
    # that's legitimately running to its cap).
    wait_budget = cfg.selfplay_worker_max_seconds + 600    # cap + margin
    nonzero = []
    for p in procs:
        try:
            p.wait(timeout=wait_budget)
        except subprocess.TimeoutExpired:
            print(f"[selfplay] worker pid={p.pid} exceeded {wait_budget}s — killing")
            p.kill()
            p.wait()
        if p.returncode not in (0, None):
            nonzero.append((p.pid, p.returncode))
    # M4: surface dead workers loudly — a CUDA-OOM race that kills some of the 7
    # workers must NOT silently train the iteration on a fraction of the data.
    if nonzero:
        print(f"[selfplay] WARNING: {len(nonzero)}/{len(procs)} workers exited "
              f"non-zero: {nonzero}")
    # match both engines' run-dir names (self_play_async / self_play_rust).
    dirs = sorted(d for d in out_dir.glob("*self_play_*") if d.is_dir())
    for d in dirs:
        # atomic meta.json tag (M2-class: a torn write would orphan the dir).
        tmp = d / "meta.json.tmp"
        tmp.write_text(json.dumps(
            {"rules_id": rules_id, "generator_name": generator_name,
             "gen_iter": gen_iter}))
        os.replace(tmp, d / "meta.json")
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
    # seed_offset=have: on a resume (have>0), shift each worker's seed-base past
    # the boards it already produced, so the deficit games are FRESH boards, not
    # replays of the existing ones (2026-06-15 resume-duplicate bug).
    dirs = _launch_selfplay_procs(cfg, Path(iter_dir) / "selfplay",
                                  Path(gen_ckpt), deficit, capped_procs,
                                  gen_name, gen_iter, cfg.rules_id,
                                  seed_offset=have)
    # M4 (deep-inspect HIGH): count THIS iteration's OWN games (prior own-iter
    # games + what we just produced) and fail loud if we fell well short of the
    # quota — instead of the binary produced==0 cliff that let a 30/1000
    # iteration pass silently after a partial worker death / OOM.
    have_after = own_iter_games(list(dirs) + list(prior_dirs), gen_iter=gen_iter)
    floor = int(cfg.selfplay_floor_ratio * cfg.games_per_iter)
    if have_after < floor:
        raise RuntimeError(
            f"self-play for iter {gen_iter} produced only {have_after} own-iter "
            f"games (quota {cfg.games_per_iter}, floor {floor}; generator "
            f"{gen_name}, ckpt {gen_ckpt}) — likely partial worker death / OOM. "
            f"Refusing to train on a fraction of the data; fix and resume.")
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
            # Match BOTH engines' run-dir names (self_play_async / self_play_rust)
            # — must mirror the tagging glob in _launch_selfplay_procs. Using the
            # async-only glob silently hid all rust prior-iter dirs, collapsing
            # the training window to one iteration (C1, code review 2026-06-19).
            dirs.extend(sorted((d for d in sp.glob("*self_play_*") if d.is_dir()),
                               reverse=True))
    return dirs


def _start_sampler(iter_dir: Path):
    """Spawn the background resource sampler writing iter_dir/resources.jsonl for
    live observability. Returns the Popen (or None if it can't start). Best
    effort — never blocks the cycle."""
    try:
        iter_dir.mkdir(parents=True, exist_ok=True)
        return subprocess.Popen(
            ["python", "-m", "catan_az.sampler", str(iter_dir), "2.0", "cycle"],
            start_new_session=True)
    except Exception:
        return None


def _stop_sampler(proc) -> None:
    if proc is None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def run_cycle(cfg, loop_root: Path, iter_n: int, capped_procs: int) -> str:
    """One full AZ cycle (redesign 2026-06-14): self-play with the LATEST
    candidate -> gen_iter window (reset on promotion) -> train/arena/publish ->
    archive. Returns the verdict. Spec 2026-06-14 §4.

    A background resource sampler runs for the whole cycle, streaming GPU/CPU/RAM
    to iter_dir/resources.jsonl for the live dashboard (observability phase 1)."""
    loop_root = Path(loop_root)
    iter_dir = loop_root / f"iter_{iter_n}"
    _sampler = _start_sampler(iter_dir)
    try:
        return _run_cycle_inner(cfg, loop_root, iter_n, capped_procs, iter_dir)
    finally:
        _stop_sampler(_sampler)


def _run_cycle_inner(cfg, loop_root: Path, iter_n: int, capped_procs: int,
                     iter_dir: Path) -> str:
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
    # Rust engine: the WHOLE driver needs the GPU env so the IN-PROCESS arena
    # (_run_arena_rust runs tch in this process) uses the GPU. LD_PRELOAD can't
    # be set after start, so re-exec once with the env if it's missing. Self-play
    # workers are subprocesses and get the env separately (_rust_cuda_env), but
    # the arena runs here -> the driver itself must carry it.
    # M2 fix (code review 2026-06-19): gate the re-exec SOLELY on the explicit
    # one-shot guard (not a fragile LD_PRELOAD substring check that could mask a
    # broken-path preload). _rust_cuda_env validates libtorch_cuda.so exists and
    # returns env without the preload if not, so we never re-exec into a bad
    # LD_PRELOAD that aborts the interpreter.
    if getattr(cfg, "engine", "python") == "rust" \
            and os.environ.get("_AZ_GPU_REEXEC") != "1":
        env = _rust_cuda_env()
        if "libtorch_cuda.so" in env.get("LD_PRELOAD", ""):
            env["_AZ_GPU_REEXEC"] = "1"
            print("[az-day] re-exec with GPU env for the rust in-process arena", flush=True)
            os.execvpe(sys.executable,
                       [sys.executable, "-m", "catan_az.daily", *sys.argv[1:]], env)
        else:
            print("[az-day] WARNING: engine=rust but libtorch_cuda.so not found — "
                  "in-process arena will run on CPU", flush=True)
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
