"""Experiment 4: round-robin-ish tournament. MCTS vs Greedy vs Random.

Seating is rotated through the 4 cyclic rotations (not all 24 permutations — that's
a lot of compute for limited statistical extra payoff for a study at this scope).
Each rotation is run `num_games_per_seating` times with different seeds.
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
from ..bots import GreedyBaselineBot
from ..evaluator import RustRolloutEvaluator
from ..recorder import SelfPlayRecorder
from .common import make_run_dir, play_one_game


class _RandomBot:
    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def step(self, state):
        return self._rng.choice(state.legal_actions())


def _build_mcts(game, sims: int, seed: int):
    rng = np.random.default_rng(seed)
    evaluator = RustRolloutEvaluator(n_rollouts=1, base_seed=seed)
    return os_mcts.MCTSBot(
        game=game, uct_c=1.4, max_simulations=sims,
        evaluator=evaluator, solve=False, random_state=rng,
    )


_BASE_SEATING = ["MCTS", "Greedy", "Random", "Random"]


def _run_cell(rec: SelfPlayRecorder, rot_idx: int, mcts_sims: int, seeds: list[int],
              done: set[int], max_seconds: float, progress_desc_prefix: str = "") -> None:
    game = CatanGame()
    seating = _BASE_SEATING[rot_idx:] + _BASE_SEATING[:rot_idx]
    desc = f"{progress_desc_prefix}rot={rot_idx} {'-'.join(seating)}"
    for seed in tqdm(seeds, desc=desc, leave=False):
        if seed in done:
            continue
        chance_rng = random.Random(seed)
        mcts_bot = _build_mcts(game, sims=mcts_sims, seed=seed)
        bots = {}
        mcts_slot = -1
        for slot, role in enumerate(seating):
            if role == "MCTS":
                bots[slot] = mcts_bot
                mcts_slot = slot
            elif role == "Greedy":
                bots[slot] = GreedyBaselineBot(seed=seed + 100 + slot)
            else:
                bots[slot] = _RandomBot(seed + 200 + slot)
        with rec.game(seed=seed) as g_rec:
            outcome = play_one_game(
                game=game, bots=bots, seed=seed, chance_rng=chance_rng,
                recorded_player=mcts_slot, recorder_game=g_rec, mcts_bot=mcts_bot,
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
                )
                rec.mark_done(seed)


def _worker(args) -> None:
    worker_idx, parent_out, mcts_sims, seeds_per_rot, max_seconds, base_config = args
    worker_dir = parent_out / f"worker{worker_idx}"
    worker_dir.mkdir(parents=True, exist_ok=True)
    rec = SelfPlayRecorder(worker_dir, config={**base_config, "worker_idx": worker_idx})
    done = rec.done_seeds()
    for rot_idx in range(4):
        seeds = seeds_per_rot[rot_idx]
        _run_cell(rec, rot_idx, mcts_sims, seeds, done, max_seconds,
                  progress_desc_prefix=f"w{worker_idx} ")
        rec.checkpoint(f"rot={rot_idx}")
    rec.flush()


def main(
    *,
    out_root: Path,
    num_games_per_seating: int = 25,
    mcts_sims: int = 100,
    seed_base: int = 4_000_000,
    max_seconds: float = 300.0,
    resume: bool = True,
    workers: int = 1,
) -> Path:
    """4 cyclic rotations of [MCTS, Greedy, Random, Random].

    v2: per-game cap, per-rotation checkpoint, resume, --workers N parallel."""
    out = make_run_dir(out_root, "e4_tournament")
    base_config = {
        "experiment": "e4_tournament",
        "mcts_sims": mcts_sims,
        "num_games_per_seating": num_games_per_seating,
        "max_seconds": max_seconds,
        "workers": workers,
    }

    if workers <= 1:
        rec = SelfPlayRecorder(out, config=base_config)
        done = rec.done_seeds() if resume else set()
        for rot_idx in range(4):
            seeds = [seed_base + rot_idx * 10_000 + i for i in range(num_games_per_seating)]
            _run_cell(rec, rot_idx, mcts_sims, seeds, done, max_seconds)
            rec.checkpoint(f"rot={rot_idx}")
        rec.flush()
        return out

    seeds_per_rot_per_worker: list[list[list[int]]] = [
        [[] for _ in range(workers)] for _ in range(4)
    ]
    for rot_idx in range(4):
        for i in range(num_games_per_seating):
            seed = seed_base + rot_idx * 10_000 + i
            seeds_per_rot_per_worker[rot_idx][i % workers].append(seed)

    args_list = []
    for w in range(workers):
        seeds_per_rot = [seeds_per_rot_per_worker[r][w] for r in range(4)]
        args_list.append((w, out, mcts_sims, seeds_per_rot, max_seconds, base_config))

    ctx = get_context("spawn")
    with ctx.Pool(processes=workers) as pool:
        pool.map(_worker, args_list)
    return out


def cli_main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", type=Path, default=Path("runs"))
    p.add_argument("--num-games-per-seating", type=int, default=25)
    p.add_argument("--mcts-sims", type=int, default=100)
    p.add_argument("--seed-base", type=int, default=4_000_000)
    p.add_argument("--max-seconds", type=float, default=300.0)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--workers", type=int, default=1)
    args = p.parse_args()
    out = main(
        out_root=args.out_root,
        num_games_per_seating=args.num_games_per_seating,
        max_seconds=args.max_seconds, resume=not args.no_resume,
        mcts_sims=args.mcts_sims, seed_base=args.seed_base,
        workers=args.workers,
    )
    print(f"e4 wrote to {out}")


if __name__ == "__main__":
    cli_main()
