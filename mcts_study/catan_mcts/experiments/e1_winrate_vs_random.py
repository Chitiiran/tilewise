"""Experiment 1: MCTS win-rate vs 3 RandomBots, sweeping simulation budget."""
from __future__ import annotations

import argparse
import random
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from open_spiel.python.algorithms import mcts as os_mcts
from tqdm import tqdm

from ..adapter import CatanGame
from ..evaluator import RustRolloutEvaluator
from ..recorder import SelfPlayRecorder
from .common import make_run_dir, play_one_game


def _build_mcts_bot(game, sims: int, seed: int):
    rng = np.random.default_rng(seed)
    # RustRolloutEvaluator pushes the per-simulation rollout into Rust
    # (~100x faster than os_mcts.RandomRolloutEvaluator on Tier-1 games).
    evaluator = RustRolloutEvaluator(n_rollouts=1, base_seed=seed)
    return os_mcts.MCTSBot(
        game=game, uct_c=1.4, max_simulations=sims,
        evaluator=evaluator, solve=False, random_state=rng,
    )


class _RandomBot:
    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def step(self, state):
        return self._rng.choice(state.legal_actions())


def _run_cell(rec: SelfPlayRecorder, sims: int, seeds: list[int],
              done: set[int], max_seconds: float, leave_progress: bool = False,
              progress_desc_prefix: str = "") -> None:
    """Run one sims-cell's worth of seeds against a fresh game/bot per seed,
    writing finalized games to `rec`'s buffers and timed-out games to skipped.csv."""
    game = CatanGame()
    desc = f"{progress_desc_prefix}sims={sims}"
    for seed in tqdm(seeds, desc=desc, leave=leave_progress):
        if seed in done:
            continue
        chance_rng = random.Random(seed)
        mcts_bot = _build_mcts_bot(game, sims=sims, seed=seed)
        bots = {0: mcts_bot, 1: _RandomBot(seed+1), 2: _RandomBot(seed+2), 3: _RandomBot(seed+3)}
        with rec.game(seed=seed) as g_rec:
            outcome = play_one_game(
                game=game, bots=bots, seed=seed, chance_rng=chance_rng,
                recorded_player=0, recorder_game=g_rec, mcts_bot=mcts_bot,
                max_seconds=max_seconds,
            )
            if outcome.timed_out:
                # v2 salvage: keep the partial action_history + moves rows so
                # we can replay/inspect the timed-out game later.
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
    """Top-level (pickle-safe) parallel worker. Runs all sims cells for one
    slice of seeds in a fresh process, writing parquet shards to a per-worker
    subdirectory."""
    worker_idx, parent_out, sims_grid, seeds_per_cell, max_seconds, base_config = args
    worker_dir = parent_out / f"worker{worker_idx}"
    worker_dir.mkdir(parents=True, exist_ok=True)
    rec = SelfPlayRecorder(worker_dir, config={**base_config, "worker_idx": worker_idx})
    done = rec.done_seeds()
    for sims, seeds in zip(sims_grid, seeds_per_cell):
        _run_cell(rec, sims, seeds, done, max_seconds,
                  progress_desc_prefix=f"w{worker_idx} ")
        rec.checkpoint(f"sims={sims}")
    rec.flush()


def main(
    *,
    out_root: Path,
    num_games: int = 50,
    sims_per_move_grid: list[int] = (5, 25, 100, 400),
    seed_base: int = 1_000_000,
    max_seconds: float = 300.0,
    resume: bool = True,
    workers: int = 1,
) -> Path:
    """v2: per-game wall-clock cap (max_seconds), per-cell parquet flush, and
    optional resume-from-done.txt so a kill+restart picks up where it left off.
    workers > 1 splits seeds across `workers` processes; each writes its own
    shards to `out_root/.../workerN/`.
    """
    out = make_run_dir(out_root, "e1_winrate_vs_random")
    base_config = {
        "experiment": "e1_winrate_vs_random",
        "uct_c": 1.4,
        "sims_per_move_grid": list(sims_per_move_grid),
        "num_games": num_games,
        "max_seconds": max_seconds,
        "workers": workers,
    }

    if workers <= 1:
        rec = SelfPlayRecorder(out, config=base_config)
        done = rec.done_seeds() if resume else set()
        for sims in sims_per_move_grid:
            seeds = [seed_base + sims * 1_000 + i for i in range(num_games)]
            _run_cell(rec, sims, seeds, done, max_seconds)
            rec.checkpoint(f"sims={sims}")
        rec.flush()
        return out

    # Parallel: split seeds round-robin across workers per cell.
    seeds_per_cell_per_worker: list[list[list[int]]] = [
        [[] for _ in range(workers)] for _ in sims_per_move_grid
    ]
    for cell_idx, sims in enumerate(sims_per_move_grid):
        for i in range(num_games):
            seed = seed_base + sims * 1_000 + i
            seeds_per_cell_per_worker[cell_idx][i % workers].append(seed)

    args_list = []
    for w in range(workers):
        seeds_per_cell = [seeds_per_cell_per_worker[c][w]
                          for c in range(len(sims_per_move_grid))]
        args_list.append(
            (w, out, list(sims_per_move_grid), seeds_per_cell, max_seconds, base_config),
        )

    ctx = get_context("spawn")
    with ctx.Pool(processes=workers) as pool:
        pool.map(_worker, args_list)

    return out


def cli_main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", type=Path, default=Path("runs"))
    p.add_argument("--num-games", type=int, default=50)
    p.add_argument("--sims-grid", type=int, nargs="+", default=[5, 25, 100, 400])
    p.add_argument("--seed-base", type=int, default=1_000_000)
    p.add_argument("--max-seconds", type=float, default=300.0,
                   help="v2: per-game wall-clock cap; timeouts go to skipped.csv")
    p.add_argument("--no-resume", action="store_true",
                   help="v2: ignore done.txt and re-run all seeds")
    p.add_argument("--workers", type=int, default=1,
                   help="v2: number of parallel worker processes (default 1 = serial)")
    args = p.parse_args()
    out = main(
        out_root=args.out_root, num_games=args.num_games,
        sims_per_move_grid=args.sims_grid, seed_base=args.seed_base,
        max_seconds=args.max_seconds, resume=not args.no_resume,
        workers=args.workers,
    )
    print(f"e1 wrote to {out}")


if __name__ == "__main__":
    cli_main()
