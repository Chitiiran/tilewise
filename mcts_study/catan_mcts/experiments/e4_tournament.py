"""Experiment 4: round-robin-ish tournament. MCTS vs Greedy vs Random.

Seating is rotated through the 4 cyclic rotations (not all 24 permutations — that's
a lot of compute for limited statistical extra payoff for a study at this scope).
Each rotation is run `num_games_per_seating` times with different seeds.
"""
from __future__ import annotations

import argparse
import random
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


def main(
    *,
    out_root: Path,
    num_games_per_seating: int = 25,
    mcts_sims: int = 100,
    seed_base: int = 4_000_000,
    max_seconds: float = 300.0,
    resume: bool = True,
) -> Path:
    """4 cyclic rotations of [MCTS, Greedy, Random, Random].

    v2: per-game cap, per-rotation checkpoint, resume support."""
    out = make_run_dir(out_root, "e4_tournament")
    rec = SelfPlayRecorder(out, config={
        "experiment": "e4_tournament",
        "mcts_sims": mcts_sims,
        "num_games_per_seating": num_games_per_seating,
        "max_seconds": max_seconds,
    })
    done = rec.done_seeds() if resume else set()

    game = CatanGame()
    base_seating = ["MCTS", "Greedy", "Random", "Random"]
    rotations = [base_seating[i:] + base_seating[:i] for i in range(4)]

    for rot_idx, seating in enumerate(rotations):
        for i in tqdm(range(num_games_per_seating), desc=f"rot={rot_idx} {'-'.join(seating)}", leave=False):
            seed = seed_base + rot_idx * 10_000 + i
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
        rec.checkpoint(f"rot={rot_idx}")
    rec.flush()
    return out


def cli_main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", type=Path, default=Path("runs"))
    p.add_argument("--num-games-per-seating", type=int, default=25)
    p.add_argument("--mcts-sims", type=int, default=100)
    p.add_argument("--seed-base", type=int, default=4_000_000)
    p.add_argument("--max-seconds", type=float, default=300.0)
    p.add_argument("--no-resume", action="store_true")
    args = p.parse_args()
    out = main(
        out_root=args.out_root,
        num_games_per_seating=args.num_games_per_seating,
        max_seconds=args.max_seconds, resume=not args.no_resume,
        mcts_sims=args.mcts_sims, seed_base=args.seed_base,
    )
    print(f"e4 wrote to {out}")


if __name__ == "__main__":
    cli_main()
