"""Experiment 2: MCTS UCB exploration constant `c` sweep at fixed simulation budget."""
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


class _RandomBot:
    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def step(self, state):
        return self._rng.choice(state.legal_actions())


def _build_mcts_bot(game, c: float, sims: int, seed: int):
    rng = np.random.default_rng(seed)
    evaluator = RustRolloutEvaluator(n_rollouts=1, base_seed=seed)
    return os_mcts.MCTSBot(
        game=game, uct_c=c, max_simulations=sims,
        evaluator=evaluator, solve=False, random_state=rng,
    )


def _run_cell(rec: SelfPlayRecorder, c: float, sims: int, seeds: list[int],
              done: set[int], max_seconds: float, progress_desc_prefix: str = "") -> None:
    """Run one c-cell's worth of seeds."""
    game = CatanGame()
    desc = f"{progress_desc_prefix}c={c}"
    for seed in tqdm(seeds, desc=desc, leave=False):
        if seed in done:
            continue
        chance_rng = random.Random(seed)
        mcts_bot = _build_mcts_bot(game, c=c, sims=sims, seed=seed)
        bots = {0: mcts_bot, 1: _RandomBot(seed+1), 2: _RandomBot(seed+2), 3: _RandomBot(seed+3)}
        with rec.game(seed=seed) as g_rec:
            outcome = play_one_game(
                game=game, bots=bots, seed=seed, chance_rng=chance_rng,
                recorded_player=0, recorder_game=g_rec, mcts_bot=mcts_bot,
                max_seconds=max_seconds,
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
    worker_idx, parent_out, c_grid, sims, seeds_per_cell, max_seconds, base_config = args
    worker_dir = parent_out / f"worker{worker_idx}"
    worker_dir.mkdir(parents=True, exist_ok=True)
    rec = SelfPlayRecorder(worker_dir, config={**base_config, "worker_idx": worker_idx})
    done = rec.done_seeds()
    for c, seeds in zip(c_grid, seeds_per_cell):
        _run_cell(rec, c, sims, seeds, done, max_seconds, progress_desc_prefix=f"w{worker_idx} ")
        rec.checkpoint(f"c={c}")
    rec.flush()


def main(
    *,
    out_root: Path,
    num_games: int = 50,
    c_grid: list[float] = (0.5, 1.0, 1.4, 2.0, 4.0),
    sims: int = 25,
    seed_base: int = 2_000_000,
    max_seconds: float = 300.0,
    resume: bool = True,
    workers: int = 1,
) -> Path:
    """v2: per-game cap, per-c-cell checkpoint, resume support, --workers N parallel."""
    out = make_run_dir(out_root, "e2_ucb_c_sweep")
    base_config = {
        "experiment": "e2_ucb_c_sweep",
        "c_grid": list(c_grid),
        "sims_per_move": sims,
        "num_games": num_games,
        "max_seconds": max_seconds,
        "workers": workers,
    }

    if workers <= 1:
        rec = SelfPlayRecorder(out, config=base_config)
        done = rec.done_seeds() if resume else set()
        for c in c_grid:
            seeds = [seed_base + int(c * 100) * 1_000 + i for i in range(num_games)]
            _run_cell(rec, c, sims, seeds, done, max_seconds)
            rec.checkpoint(f"c={c}")
        rec.flush()
        return out

    seeds_per_cell_per_worker: list[list[list[int]]] = [
        [[] for _ in range(workers)] for _ in c_grid
    ]
    for cell_idx, c in enumerate(c_grid):
        for i in range(num_games):
            seed = seed_base + int(c * 100) * 1_000 + i
            seeds_per_cell_per_worker[cell_idx][i % workers].append(seed)

    args_list = []
    for w in range(workers):
        seeds_per_cell = [seeds_per_cell_per_worker[c_idx][w] for c_idx in range(len(c_grid))]
        args_list.append((w, out, list(c_grid), sims, seeds_per_cell, max_seconds, base_config))

    ctx = get_context("spawn")
    with ctx.Pool(processes=workers) as pool:
        pool.map(_worker, args_list)
    return out


def cli_main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", type=Path, default=Path("runs"))
    p.add_argument("--num-games", type=int, default=50)
    p.add_argument("--c-grid", type=float, nargs="+", default=[0.5, 1.0, 1.4, 2.0, 4.0])
    p.add_argument("--sims", type=int, default=25)
    p.add_argument("--seed-base", type=int, default=2_000_000)
    p.add_argument("--max-seconds", type=float, default=300.0)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--workers", type=int, default=1)
    args = p.parse_args()
    out = main(
        out_root=args.out_root, num_games=args.num_games,
        c_grid=args.c_grid, sims=args.sims, seed_base=args.seed_base,
        max_seconds=args.max_seconds, resume=not args.no_resume,
        workers=args.workers,
    )
    print(f"e2 wrote to {out}")


if __name__ == "__main__":
    cli_main()
