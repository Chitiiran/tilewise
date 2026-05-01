"""Experiment 3: MCTS rollout policy comparison — random vs heuristic, across sim budgets."""
from __future__ import annotations

import argparse
import random
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from open_spiel.python.algorithms import mcts as os_mcts
from tqdm import tqdm

from ..adapter import CatanGame
from ..bots import heuristic_rollout
from ..evaluator import RustRolloutEvaluator
from ..recorder import SelfPlayRecorder
from .common import make_run_dir, play_one_game


class _RandomBot:
    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def step(self, state):
        return self._rng.choice(state.legal_actions())


class _HeuristicEvaluator(os_mcts.Evaluator):
    """OpenSpiel evaluator that rolls out using `heuristic_rollout`."""

    def __init__(self, n_rollouts: int, rng: random.Random) -> None:
        self._n = n_rollouts
        self._rng = rng

    def evaluate(self, state):
        wins = np.zeros(state.num_players(), dtype=np.float32)
        for _ in range(self._n):
            sim = state.clone()
            steps = 0
            while not sim.is_terminal() and steps < 30000:
                if sim.is_chance_node():
                    outcomes = sim.chance_outcomes()
                    r = self._rng.random()
                    cum, chosen = 0.0, outcomes[-1][0]
                    for v, p in outcomes:
                        cum += p
                        if r <= cum:
                            chosen = v
                            break
                    sim.apply_action(int(chosen))
                else:
                    sim.apply_action(int(heuristic_rollout(sim, self._rng)))
                steps += 1
            wins += np.array(sim.returns(), dtype=np.float32)
        return wins / self._n

    def prior(self, state):
        legal = state.legal_actions()
        p = 1.0 / len(legal)
        return [(a, p) for a in legal]


def _bot(game, sims: int, evaluator, seed: int):
    rng = np.random.default_rng(seed)
    return os_mcts.MCTSBot(
        game=game, uct_c=1.4, max_simulations=sims,
        evaluator=evaluator, solve=False, random_state=rng,
    )


def _run_cell(rec: SelfPlayRecorder, policy_name: str, sims: int, seeds: list[int],
              done: set[int], max_seconds: float, progress_desc_prefix: str = "") -> None:
    game = CatanGame()
    desc = f"{progress_desc_prefix}{policy_name} sims={sims}"
    for seed in tqdm(seeds, desc=desc, leave=False):
        if seed in done:
            continue
        chance_rng = random.Random(seed)
        if policy_name == "random":
            evaluator = RustRolloutEvaluator(n_rollouts=1, base_seed=seed)
        else:
            evaluator = _HeuristicEvaluator(n_rollouts=1, rng=random.Random(seed))
        mcts_bot = _bot(game, sims=sims, evaluator=evaluator, seed=seed)
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


def _seed_for(seed_base: int, policy_name: str, sims: int, i: int) -> int:
    return seed_base + (0 if policy_name == "random" else 1) * 1_000_000 + sims * 1_000 + i


def _worker(args) -> None:
    worker_idx, parent_out, sims_grid, seeds_per_cell, max_seconds, base_config = args
    worker_dir = parent_out / f"worker{worker_idx}"
    worker_dir.mkdir(parents=True, exist_ok=True)
    rec = SelfPlayRecorder(worker_dir, config={**base_config, "worker_idx": worker_idx})
    done = rec.done_seeds()
    cell_idx = 0
    for policy_name in ("random", "heuristic"):
        for sims in sims_grid:
            seeds = seeds_per_cell[cell_idx]
            cell_idx += 1
            _run_cell(rec, policy_name, sims, seeds, done, max_seconds,
                      progress_desc_prefix=f"w{worker_idx} ")
            rec.checkpoint(f"{policy_name}-sims={sims}")
    rec.flush()


def main(
    *,
    out_root: Path,
    num_games: int = 50,
    sims_grid: list[int] = (5, 25, 100),
    seed_base: int = 3_000_000,
    max_seconds: float = 300.0,
    resume: bool = True,
    workers: int = 1,
) -> Path:
    """v2: per-game cap, per-(policy,sims)-cell checkpoint, resume, --workers N."""
    out = make_run_dir(out_root, "e3_rollout_policy")
    base_config = {
        "experiment": "e3_rollout_policy",
        "uct_c": 1.4,
        "sims_grid": list(sims_grid),
        "num_games": num_games,
        "rollout_policies": ["random", "heuristic"],
        "max_seconds": max_seconds,
        "workers": workers,
    }

    if workers <= 1:
        rec = SelfPlayRecorder(out, config=base_config)
        done = rec.done_seeds() if resume else set()
        for policy_name in ("random", "heuristic"):
            for sims in sims_grid:
                seeds = [_seed_for(seed_base, policy_name, sims, i) for i in range(num_games)]
                _run_cell(rec, policy_name, sims, seeds, done, max_seconds)
                rec.checkpoint(f"{policy_name}-sims={sims}")
        rec.flush()
        return out

    cells = [(p, s) for p in ("random", "heuristic") for s in sims_grid]
    seeds_per_cell_per_worker: list[list[list[int]]] = [
        [[] for _ in range(workers)] for _ in cells
    ]
    for cell_idx, (policy_name, sims) in enumerate(cells):
        for i in range(num_games):
            seed = _seed_for(seed_base, policy_name, sims, i)
            seeds_per_cell_per_worker[cell_idx][i % workers].append(seed)

    args_list = []
    for w in range(workers):
        seeds_per_cell = [seeds_per_cell_per_worker[c][w] for c in range(len(cells))]
        args_list.append((w, out, list(sims_grid), seeds_per_cell, max_seconds, base_config))

    ctx = get_context("spawn")
    with ctx.Pool(processes=workers) as pool:
        pool.map(_worker, args_list)
    return out


def cli_main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", type=Path, default=Path("runs"))
    p.add_argument("--num-games", type=int, default=50)
    p.add_argument("--sims-grid", type=int, nargs="+", default=[5, 25, 100])
    p.add_argument("--seed-base", type=int, default=3_000_000)
    p.add_argument("--max-seconds", type=float, default=300.0)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--workers", type=int, default=1)
    args = p.parse_args()
    out = main(
        out_root=args.out_root, num_games=args.num_games,
        sims_grid=args.sims_grid, seed_base=args.seed_base,
        max_seconds=args.max_seconds, resume=not args.no_resume,
        workers=args.workers,
    )
    print(f"e3 wrote to {out}")


if __name__ == "__main__":
    cli_main()
