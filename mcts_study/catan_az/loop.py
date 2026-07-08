"""AZ loop orchestrator (spec §2).

run_iteration(...) drives SELF-PLAY -> TRAIN -> ARENA -> PUBLISH for one
iteration, with stage functions injected so the bookkeeping is testable
with fakes. Defaults wrap the proven pieces (self_play_async, train_main,
run_arena).

Resumability: each completed stage drops <iter_dir>/<STAGE>.done with its
output paths; a rerun reuses those instead of recomputing. An ARENA that
returns "invalid" (timeout-censored) does NOT mark done — it must rerun.
A STOP sentinel in loop_root halts cleanly between stages.

CLI:
  python -m catan_az.loop --loop-root R --iter 1 \
      --skip-selfplay-dirs d1 d2 ...   # consume pre-existing self-play dirs
  python -m catan_az.loop --loop-root R --forever
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .arena import ArenaResult, run_arena, should_promote
from .buffer import select_window
from .config import AzConfig
from .data_quality import degeneracy_verdict, summarize_selfplay_dir
from .ladder import Ladder
from .status import StatusWriter


class StopRequested(Exception):
    pass


def _check_stop(loop_root: Path) -> None:
    if (loop_root / "STOP").exists():
        raise StopRequested(f"STOP sentinel present in {loop_root}")


def _done_marker(iter_dir: Path, stage: str) -> Path:
    return iter_dir / f"{stage}.done"


def _mark_done(iter_dir: Path, stage: str, payload: dict) -> None:
    _done_marker(iter_dir, stage).write_text(json.dumps(payload))


def _load_done(iter_dir: Path, stage: str) -> dict | None:
    m = _done_marker(iter_dir, stage)
    return json.loads(m.read_text()) if m.exists() else None


def _check_data_quality(cfg: AzConfig, iter_dir: Path, run_dirs: list[Path]) -> None:
    """Summarize this iteration's freshly-generated self-play dirs, write
    data_quality.json next to SELFPLAY.done, and refuse to mark the stage
    healthy (raise) when the data is degenerate (Task 7, observability
    minimum — shake-out journal 2026-06-19 §3: an all-timeout self-play run
    was invisible until training wasted hours on it).

    Aggregates ACROSS all of this iteration's run_dirs (a single verdict for
    the iteration, not per-dir) — self-play launches multiple worker dirs for
    one logical iteration, and a per-dir verdict would let one degenerate
    worker hide behind healthy siblings' totals or vice versa.

    Deliberately NOT applied to the existing_selfplay_dirs salvage path: that
    branch consumes pre-existing, already-produced data by explicit operator
    choice (e.g. iteration-1 salvage) — gating it here would make the loop
    refuse data the operator already decided to use, which is a different
    control point than this fresh-self-play stage.
    """
    games = 0
    winners_by_seat = {"0": 0, "1": 0, "2": 0, "3": 0}
    timeouts = 0
    no_winner = 0
    length_max = 0
    for d in run_dirs:
        s = summarize_selfplay_dir(d)
        games += s["games"]
        for k in winners_by_seat:
            winners_by_seat[k] += s["winners_by_seat"].get(k, 0)
        timeouts += s["timeouts"]
        no_winner += s["no_winner"]
        length_max = max(length_max, s["length_max"])
    summary = {
        "games": games,
        "winners_by_seat": winners_by_seat,
        "timeouts": timeouts,
        "no_winner": no_winner,
        "length_max": length_max,
    }
    verdict = degeneracy_verdict(summary, cfg)
    summary["verdict"] = verdict
    (iter_dir / "data_quality.json").write_text(json.dumps(summary, indent=2))
    if verdict == "degenerate":
        raise RuntimeError(
            f"self-play for {iter_dir} produced degenerate data ({games} games, "
            f"{timeouts} timeouts, {no_winner} no-winner) — refusing to mark the "
            f"stage healthy / train on it. See {iter_dir / 'data_quality.json'}.")


# -- default stage implementations ------------------------------------------

def _default_selfplay(cfg: AzConfig, iter_dir: Path, champion_ckpt: str) -> list[Path]:
    """Single-process default; the 5-proc launch stays a shell concern
    (scripts/distill_runbook.md pattern) until the B1 spike lands."""
    from catan_mcts.experiments.self_play_async import run_self_play
    out = run_self_play(
        out_root=iter_dir, checkpoint=Path(champion_ckpt),
        num_games=cfg.games_per_iter, n_sims=cfg.sims,
        n_concurrent=cfg.n_concurrent, hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers, vp_target=cfg.vp_target,
        bonuses=cfg.bonuses, device="cuda", max_batch=cfg.max_batch,
        max_seconds=6 * 3600, seed_base=31_000_000, self_play=True)
    return [out]


def _default_train(cfg: AzConfig, run_dirs: list[Path], iter_dir: Path,
                   init_ckpt: str) -> Path:
    from catan_gnn.train import train_main
    out_dir = iter_dir / "training"
    train_main(
        run_dirs=run_dirs, out_dir=out_dir,
        hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers,
        epochs=cfg.max_epochs, batch_size=cfg.batch_size, lr=cfg.lr,
        init_from=Path(init_ckpt), policy_sharpen=cfg.policy_sharpen,
        early_stop_patience=1 if cfg.early_stop else 0,
        device="cuda")
    best = out_dir / "checkpoint_best.pt"
    if not best.exists():
        raise FileNotFoundError(f"training produced no {best}")
    # TorchScript export for the Rust self-play/arena engines. MUST match the
    # device-suffixed paths the consumers load (self_play_rust._ensure_ts /
    # _ensure_batched_ts, arena._ts) AND be traced on the inference device — a
    # plain CPU-traced .ts is both orphaned (nobody reads it) and crashes on
    # CUDA (H1, code review 2026-06-19). Export both single + batched on-device
    # so consumers reuse them instead of re-tracing every launch.
    import torch
    from catan_gnn.export_torchscript import export, export_batched
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bmax = getattr(cfg, "max_batch", 32)
    export(checkpoint=best, out_ts=best.with_suffix(f".{dev}.ts"),
           hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers, device=dev)
    # Match self_play_rust._ensure_batched_ts naming (.{dev}.b{bmax}.batch.ts) so
    # the export is REUSED, not re-traced per worker launch.
    export_batched(checkpoint=best, out_ts=best.with_suffix(f".{dev}.b{bmax}.batch.ts"),
                   hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers,
                   b_max=bmax, device=dev)
    return best


def _default_arena(cfg: AzConfig, cand_ckpt: Path, champ_ckpt: str,
                   iter_dir: Path) -> ArenaResult:
    return run_arena(candidate_ckpt=Path(cand_ckpt),
                     champion_ckpt=Path(champ_ckpt), cfg=cfg,
                     out_dir=iter_dir / "arena")


# -- orchestration -----------------------------------------------------------

def run_iteration(cfg: AzConfig, loop_root: Path, iter_n: int, *,
                  selfplay_fn=_default_selfplay, train_fn=_default_train,
                  arena_fn=_default_arena,
                  existing_selfplay_dirs: list[Path] | None = None) -> str:
    """Run one iteration; returns the arena verdict ('promote'|'hold'|'invalid')."""
    loop_root = Path(loop_root)
    iter_dir = loop_root / f"iter_{iter_n}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    status = StatusWriter(loop_root)
    ladder = Ladder(loop_root)
    champion = ladder.champion()

    # SELF-PLAY
    _check_stop(loop_root)
    done = _load_done(iter_dir, "SELFPLAY")
    if done:
        run_dirs = [Path(p) for p in done["run_dirs"]]
    elif existing_selfplay_dirs is not None:
        run_dirs = [Path(p) for p in existing_selfplay_dirs]
        _mark_done(iter_dir, "SELFPLAY", {"run_dirs": [str(p) for p in run_dirs],
                                          "source": "existing"})
    else:
        status.stage(iter_n, "selfplay", champion=champion["name"])
        run_dirs = selfplay_fn(cfg, iter_dir, champion["checkpoint"])
        # Degeneracy gate BEFORE the done-marker: a degenerate run must not be
        # remembered as "self-play complete" — a rerun should retry self-play,
        # not resume past it (RuntimeError below prevents _mark_done from
        # ever executing on the degenerate path).
        _check_data_quality(cfg, iter_dir, run_dirs)
        _mark_done(iter_dir, "SELFPLAY", {"run_dirs": [str(p) for p in run_dirs]})

    # BUFFER: window over this + previous iterations' self-play dirs.
    # When run_cycle hands us the accumulating window (gen_window over
    # gen_iter>=last_promotion_iter), HONOR IT — do NOT re-derive + re-truncate
    # to cfg.window_games. That second select_window silently capped the
    # design's accumulating window (2786 games) back to 1200 (2026-06-17 bug).
    #
    # CRUCIAL: the window comes from existing_selfplay_dirs (the accumulating
    # set), NOT from run_dirs. On a RESUME run_dirs is loaded from SELFPLAY.done,
    # which lists only THIS iteration's own self-play dirs (for not regenerating
    # games) — using it for the window would shadow the accumulating set and
    # train on just this iter's ~945 games (2026-06-17 resume-window bug). The
    # SELFPLAY marker governs self-play-resume; the WINDOW must follow the
    # accumulating dirs run_cycle computed.
    if existing_selfplay_dirs is not None:
        window = [Path(p) for p in existing_selfplay_dirs]
    else:
        all_dirs_newest_first = run_dirs + _previous_selfplay_dirs(loop_root, iter_n)
        window = select_window(all_dirs_newest_first, cfg.window_games)

    # TRAIN
    _check_stop(loop_root)
    done = _load_done(iter_dir, "TRAIN")
    if done:
        candidate = Path(done["candidate"])
    else:
        status.stage(iter_n, "train", window_dirs=len(window))
        candidate = train_fn(cfg, window, iter_dir, champion["checkpoint"])
        _mark_done(iter_dir, "TRAIN", {"candidate": str(candidate)})

    # ARENA
    _check_stop(loop_root)
    done = _load_done(iter_dir, "ARENA")
    if done:
        result = ArenaResult(**done["result"])
        verdict = done["verdict"]
    else:
        status.stage(iter_n, "arena", candidate=str(candidate))
        result = arena_fn(cfg, candidate, champion["checkpoint"], iter_dir)
        verdict = should_promote(result, cfg)
        if verdict != "invalid":
            _mark_done(iter_dir, "ARENA", {
                "result": {"wins_cand": result.wins_cand,
                           "wins_champ": result.wins_champ,
                           "draws": result.draws, "timeouts": result.timeouts,
                           "per_rotation": result.per_rotation},
                "verdict": verdict})

    # PUBLISH — guarded by its own done-marker so a resumed/re-run iteration
    # never double-counts Elo or appends a duplicate journal row (2026-06-13:
    # a restart-to-unblock re-ran publish, giving az_iter_1 240 games + a
    # duplicate promote). The ladder mutations + journal_row are NOT
    # individually idempotent, so the whole stage must run at most once.
    if _load_done(iter_dir, "PUBLISH") is None:
        name = f"az_iter_{iter_n}"
        if verdict == "promote":
            ckpt_dir = loop_root / "checkpoints"
            ckpt_dir.mkdir(exist_ok=True)
            published = ckpt_dir / f"{name}.pt"
            shutil.copyfile(candidate, published)
            ladder.register_candidate(name, str(published), created_iter=iter_n)
            ladder.record_arena(name, champion["name"], wins_a=result.wins_cand,
                                wins_b=result.wins_champ, draws=result.draws)
            ladder.promote(name)
        elif verdict == "hold":
            ladder.register_candidate(name, str(candidate), created_iter=iter_n)
            ladder.record_arena(name, champion["name"], wins_a=result.wins_cand,
                                wins_b=result.wins_champ, draws=result.draws)

        status.journal_row({
            "iter": iter_n, "champion": champion["name"],
            "selfplay_dirs": len(run_dirs), "window_dirs": len(window),
            "arena_wins_cand": result.wins_cand,
            "arena_wins_champ": result.wins_champ,
            "arena_draws": result.draws, "arena_timeouts": result.timeouts,
            "arena_winrate": round(result.winrate_cand, 4),
            "verdict": verdict,
            "champion_elo_after": Ladder(loop_root).champion()["elo"],
        })
        _mark_done(iter_dir, "PUBLISH", {"verdict": verdict})

    status.stage(iter_n, "done", verdict=verdict)
    return verdict


def _previous_selfplay_dirs(loop_root: Path, iter_n: int) -> list[Path]:
    """Self-play dirs of earlier iterations, newest first, from done markers."""
    dirs: list[Path] = []
    for n in range(iter_n - 1, 0, -1):
        done = _load_done(loop_root / f"iter_{n}", "SELFPLAY")
        if done:
            dirs.extend(Path(p) for p in done["run_dirs"])
    return dirs


def run_forever(cfg: AzConfig, loop_root: Path, start_iter: int = 1) -> None:
    n = start_iter
    while True:
        try:
            verdict = run_iteration(cfg, loop_root, n)
        except StopRequested:
            print(f"[az_loop] STOP honored before iter {n} completed")
            return
        print(f"[az_loop] iter {n}: {verdict}")
        n += 1


def cli_main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--loop-root", type=Path, required=True)
    p.add_argument("--config", type=Path, default=None,
                   help="AzConfig json; defaults to spec defaults")
    p.add_argument("--iter", type=int, default=None)
    p.add_argument("--forever", action="store_true")
    p.add_argument("--skip-selfplay-dirs", type=Path, nargs="*", default=None,
                   help="consume these pre-existing self-play run dirs as this "
                        "iteration's SELF-PLAY output (iteration-1 salvage)")
    args = p.parse_args()
    cfg = AzConfig.from_json(args.config) if args.config else AzConfig()
    if args.forever:
        run_forever(cfg, args.loop_root)
    elif args.iter is not None:
        verdict = run_iteration(cfg, args.loop_root, args.iter,
                                existing_selfplay_dirs=args.skip_selfplay_dirs)
        print(f"[az_loop] iter {args.iter}: {verdict}")
    else:
        p.error("need --iter N or --forever")


if __name__ == "__main__":
    cli_main()
