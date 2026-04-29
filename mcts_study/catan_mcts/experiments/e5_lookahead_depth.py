"""Experiment 5: lookahead-VP evaluator depth sweep.

Replaces random rollouts with a depth-bounded greedy-VP lookahead. Compared to
e1/e2/e3 (which all use the random-rollout evaluator), this gives MCTS a stable,
signed value for every leaf — no more [0,0,0,0] safety-cap returns swamping the
search. The cost is bias: the lookahead is greedy, so MCTS becomes a kind of
"greedy-augmented" search rather than pure rollout-Monte-Carlo. The hope is that
removing the noise pays for the bias.

Cells:
    depth in {3, 10, 25}, sims in {25, 100, 400}
    -> 9 cells × num_games games each.

Comparable to e1 (random rollouts at sims in {5,25,100}) for the rollout-vs-
lookahead writeup. The skip-trivial-turn optimization in play_one_game is also
in play, so per-game wall-clock will be substantially shorter than e1 even at
matched sims.
"""
from __future__ import annotations

import argparse
import random
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from open_spiel.python.algorithms import mcts as os_mcts
from tqdm import tqdm

from ..adapter import CatanGame
from ..evaluator import LookaheadVpEvaluator
from ..recorder import SelfPlayRecorder
from .common import make_run_dir, play_one_game


class _RandomBot:
    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def step(self, state):
        return self._rng.choice(state.legal_actions())


def _build_mcts(game, sims: int, depth: int, seed: int):
    rng = np.random.default_rng(seed)
    evaluator = LookaheadVpEvaluator(depth=depth, base_seed=seed)
    return os_mcts.MCTSBot(
        game=game, uct_c=1.4, max_simulations=sims,
        evaluator=evaluator, solve=False, random_state=rng,
    )


def _run_cell(rec: SelfPlayRecorder, depth: int, sims: int, seeds: list[int],
              done: set[int], max_seconds: float, progress_desc_prefix: str = "") -> None:
    game = CatanGame()
    desc = f"{progress_desc_prefix}depth={depth} sims={sims}"
    for seed in tqdm(seeds, desc=desc, leave=False):
        if seed in done:
            continue
        chance_rng = random.Random(seed)
        mcts_bot = _build_mcts(game, sims=sims, depth=depth, seed=seed)
        bots = {0: mcts_bot, 1: _RandomBot(seed+1), 2: _RandomBot(seed+2), 3: _RandomBot(seed+3)}
        with rec.game(seed=seed) as g_rec:
            outcome = play_one_game(
                game=game, bots=bots, seed=seed, chance_rng=chance_rng,
                recorded_player=0, recorder_game=g_rec, mcts_bot=mcts_bot,
                max_seconds=max_seconds,
            )
            if outcome.timed_out:
                g_rec._moves.clear()
                g_rec._finalized = True
                rec.skip_game(
                    seed=seed, reason="wall-clock-timeout",
                    length_in_moves=outcome.length_in_moves,
                )
            else:
                g_rec.finalize(
                    winner=outcome.winner,
                    final_vp=outcome.final_vp,
                    length_in_moves=outcome.length_in_moves,
                    action_history=outcome.action_history,
                )
                rec.mark_done(seed)


def _seed_for(seed_base: int, depth: int, sims: int, i: int) -> int:
    # depth in 0..63, sims in 0..16383 -> distinct seeds across the 9-cell grid.
    return seed_base + depth * 1_000_000 + sims * 1_000 + i


def _worker(args) -> None:
    worker_idx, parent_out, depth_grid, sims_grid, seeds_per_cell, max_seconds, base_config = args
    worker_dir = parent_out / f"worker{worker_idx}"
    worker_dir.mkdir(parents=True, exist_ok=True)
    rec = SelfPlayRecorder(worker_dir, config={**base_config, "worker_idx": worker_idx})
    done = rec.done_seeds()
    cell_idx = 0
    for depth in depth_grid:
        for sims in sims_grid:
            seeds = seeds_per_cell[cell_idx]
            cell_idx += 1
            _run_cell(rec, depth, sims, seeds, done, max_seconds,
                      progress_desc_prefix=f"w{worker_idx} ")
            rec.checkpoint(f"depth={depth}-sims={sims}")
    rec.flush()


def main(
    *,
    out_root: Path,
    num_games: int = 10,
    depth_grid: list[int] = (3, 10, 25),
    sims_grid: list[int] = (25, 100, 400),
    seed_base: int = 5_000_000,
    max_seconds: float = 300.0,
    resume: bool = True,
    workers: int = 1,
) -> Path:
    """v2: per-game cap, per-cell checkpoint, resume support, --workers N parallel."""
    out = make_run_dir(out_root, "e5_lookahead_depth")
    base_config = {
        "experiment": "e5_lookahead_depth",
        "uct_c": 1.4,
        "depth_grid": list(depth_grid),
        "sims_grid": list(sims_grid),
        "num_games": num_games,
        "evaluator": "LookaheadVpEvaluator",
        "max_seconds": max_seconds,
        "workers": workers,
    }

    if workers <= 1:
        rec = SelfPlayRecorder(out, config=base_config)
        done = rec.done_seeds() if resume else set()
        for depth in depth_grid:
            for sims in sims_grid:
                seeds = [_seed_for(seed_base, depth, sims, i) for i in range(num_games)]
                _run_cell(rec, depth, sims, seeds, done, max_seconds)
                rec.checkpoint(f"depth={depth}-sims={sims}")
        rec.flush()
        return out

    cells = [(d, s) for d in depth_grid for s in sims_grid]
    seeds_per_cell_per_worker: list[list[list[int]]] = [
        [[] for _ in range(workers)] for _ in cells
    ]
    for cell_idx, (depth, sims) in enumerate(cells):
        for i in range(num_games):
            seed = _seed_for(seed_base, depth, sims, i)
            seeds_per_cell_per_worker[cell_idx][i % workers].append(seed)

    args_list = []
    for w in range(workers):
        seeds_per_cell = [seeds_per_cell_per_worker[c][w] for c in range(len(cells))]
        args_list.append((
            w, out, list(depth_grid), list(sims_grid),
            seeds_per_cell, max_seconds, base_config,
        ))

    ctx = get_context("spawn")
    with ctx.Pool(processes=workers) as pool:
        pool.map(_worker, args_list)
    return out


def cli_main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", type=Path, default=Path("runs"))
    p.add_argument("--num-games", type=int, default=10)
    p.add_argument("--depth-grid", type=int, nargs="+", default=[3, 10, 25])
    p.add_argument("--sims-grid", type=int, nargs="+", default=[25, 100, 400])
    p.add_argument("--seed-base", type=int, default=5_000_000)
    p.add_argument("--max-seconds", type=float, default=300.0)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--workers", type=int, default=1)
    args = p.parse_args()
    out = main(
        out_root=args.out_root, num_games=args.num_games,
        depth_grid=args.depth_grid, sims_grid=args.sims_grid,
        seed_base=args.seed_base, max_seconds=args.max_seconds,
        resume=not args.no_resume, workers=args.workers,
    )
    print(f"e5 wrote to {out}")


if __name__ == "__main__":
    cli_main()
