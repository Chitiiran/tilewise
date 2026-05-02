"""Experiment 9: v3 Catan-Lite data generation.

All four seats play with `LookaheadMctsV3` (per spec §5: per-player VP-aware
exponential sim schedule on top of `LookaheadVpEvaluator`). The engine runs
in v3 mode (vp_target=5, bonuses=False).

Per game we record ONE seat's MCTS decisions to keep the parquet schema
identical to v2 runs (one (seed, recorded_player) tuple per game). The
recorded seat rotates across seeds: seed i records seat i % 4, so a
balanced sweep of N games produces N/4 positions per perspective.

Recorder writes:
  out/games.<timestamp>.parquet  — one row per game (seed, winner, action_history, ...)
  out/moves.<timestamp>.parquet  — one row per recorded MCTS decision

These feed straight into `CatanReplayDataset` (no schema change required —
v3 flags travel via the engine's deterministic state).

Resume + per-cell flush + multi-worker pool follow the same recipes used in
e5 / e7 (see memory `project_mcts_concrete_implementation_recipes`).
"""
from __future__ import annotations

import argparse
import random
from multiprocessing import get_context
from pathlib import Path

from tqdm import tqdm

from ..adapter import CatanGame
from ..players_v3 import build_lookahead_mcts_v3
from ..recorder import SelfPlayRecorder
from .common import make_run_dir, play_one_game


def _build_bots(game: CatanGame, seed: int, base_sims: int, lookahead_depth: int) -> dict:
    """Four LookaheadMctsV3 bots, one per seat. Distinct seeds per seat."""
    return {
        p: build_lookahead_mcts_v3(
            game, lookahead_depth=lookahead_depth, seed=seed + p, base_sims=base_sims,
        )
        for p in range(4)
    }


def _run_cell(
    rec: SelfPlayRecorder,
    seeds: list[int],
    done: set[int],
    *,
    base_sims: int,
    lookahead_depth: int,
    max_seconds: float,
    vp_target: int,
    bonuses: bool,
    progress_desc_prefix: str = "",
) -> None:
    game = CatanGame(vp_target=vp_target, bonuses=bonuses)
    desc = f"{progress_desc_prefix}base_sims={base_sims} depth={lookahead_depth}"
    for seed in tqdm(seeds, desc=desc, leave=False):
        if seed in done:
            continue
        chance_rng = random.Random(seed)
        bots = _build_bots(game, seed, base_sims, lookahead_depth)
        recorded_player = seed % 4
        mcts_bot = bots[recorded_player]
        with rec.game(seed=seed) as g_rec:
            outcome = play_one_game(
                game=game, bots=bots, seed=seed, chance_rng=chance_rng,
                recorded_player=recorded_player, recorder_game=g_rec,
                mcts_bot=mcts_bot, max_seconds=max_seconds,
            )
            if outcome.timed_out:
                rec.skip_game(
                    seed=seed, reason="wall-clock-timeout",
                    length_in_moves=outcome.length_in_moves,
                    action_history=outcome.action_history,
                    moves_recorder=g_rec,
                )
            else:
                g_rec.finalize(
                    winner=outcome.winner,
                    final_vp=outcome.final_vp,
                    length_in_moves=outcome.length_in_moves,
                    action_history=outcome.action_history,
                )
                rec.mark_done(seed)


def _worker(args) -> None:
    (
        worker_idx, parent_out, seeds, base_sims, lookahead_depth,
        max_seconds, vp_target, bonuses, base_config,
    ) = args
    worker_dir = parent_out / f"worker{worker_idx}"
    worker_dir.mkdir(parents=True, exist_ok=True)
    rec = SelfPlayRecorder(
        worker_dir, config={**base_config, "worker_idx": worker_idx},
    )
    done = rec.done_seeds()
    _run_cell(
        rec, seeds, done,
        base_sims=base_sims, lookahead_depth=lookahead_depth,
        max_seconds=max_seconds, vp_target=vp_target, bonuses=bonuses,
        progress_desc_prefix=f"w{worker_idx} ",
    )
    rec.checkpoint("v3-final")
    rec.flush()


def main(
    *,
    out_root: Path,
    num_games: int = 1000,
    base_sims: int = 200,
    lookahead_depth: int = 10,
    seed_base: int = 9_000_000,
    max_seconds: float = 600.0,
    vp_target: int = 5,
    bonuses: bool = False,
    resume: bool = True,
    workers: int = 1,
) -> Path:
    out = make_run_dir(out_root, "e9_v3_data_gen")
    base_config = {
        "experiment": "e9_v3_data_gen",
        "uct_c": 1.4,
        "base_sims": base_sims,
        "lookahead_depth": lookahead_depth,
        "num_games": num_games,
        "evaluator": "LookaheadMctsV3",
        "max_seconds": max_seconds,
        "vp_target": vp_target,
        "bonuses": bonuses,
        "workers": workers,
    }

    seeds_all = [seed_base + i for i in range(num_games)]

    if workers <= 1:
        rec = SelfPlayRecorder(out, config=base_config)
        done = rec.done_seeds() if resume else set()
        _run_cell(
            rec, seeds_all, done,
            base_sims=base_sims, lookahead_depth=lookahead_depth,
            max_seconds=max_seconds, vp_target=vp_target, bonuses=bonuses,
        )
        rec.checkpoint("v3-final")
        rec.flush()
        return out

    # Round-robin seeds across workers so each gets a balanced mix of
    # recorded-player rotations (since recorded_player = seed % 4).
    seeds_per_worker: list[list[int]] = [[] for _ in range(workers)]
    for i, s in enumerate(seeds_all):
        seeds_per_worker[i % workers].append(s)

    args_list = [
        (
            w, out, seeds_per_worker[w], base_sims, lookahead_depth,
            max_seconds, vp_target, bonuses, base_config,
        )
        for w in range(workers)
    ]
    ctx = get_context("spawn")
    with ctx.Pool(processes=workers) as pool:
        pool.map(_worker, args_list)
    return out


def cli_main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", type=Path, default=Path("runs"))
    p.add_argument("--num-games", type=int, default=1000)
    p.add_argument("--base-sims", type=int, default=200,
                   help="MCTS sim budget at acting_vp=0 (decays per spec §5)")
    p.add_argument("--lookahead-depth", type=int, default=10)
    p.add_argument("--seed-base", type=int, default=9_000_000)
    p.add_argument("--max-seconds", type=float, default=600.0,
                   help="Per-game wall-clock cap; default 10 min")
    p.add_argument("--vp-target", type=int, default=5)
    p.add_argument("--bonuses", action="store_true",
                   help="Enable +2 LR/LA bonuses (default off in v3)")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--workers", type=int, default=1)
    args = p.parse_args()
    out = main(
        out_root=args.out_root,
        num_games=args.num_games,
        base_sims=args.base_sims,
        lookahead_depth=args.lookahead_depth,
        seed_base=args.seed_base,
        max_seconds=args.max_seconds,
        vp_target=args.vp_target,
        bonuses=args.bonuses,
        resume=not args.no_resume,
        workers=args.workers,
    )
    print(f"e9 wrote to {out}")


if __name__ == "__main__":
    cli_main()
