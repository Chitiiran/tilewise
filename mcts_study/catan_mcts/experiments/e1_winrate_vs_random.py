"""Experiment 1: MCTS win-rate vs 3 RandomBots, sweeping simulation budget."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from open_spiel.python.algorithms import mcts as os_mcts
from tqdm import tqdm

from ..adapter import CatanGame
from ..recorder import SelfPlayRecorder
from .common import make_run_dir, play_one_game


def _build_mcts_bot(game, sims: int, seed: int):
    rng = np.random.default_rng(seed)
    evaluator = os_mcts.RandomRolloutEvaluator(n_rollouts=1, random_state=rng)
    return os_mcts.MCTSBot(
        game=game, uct_c=1.4, max_simulations=sims,
        evaluator=evaluator, solve=False, random_state=rng,
    )


class _RandomBot:
    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def step(self, state):
        return self._rng.choice(state.legal_actions())


def main(
    *,
    out_root: Path,
    num_games: int = 50,
    sims_per_move_grid: list[int] = (5, 25, 100, 400),
    seed_base: int = 1_000_000,
) -> Path:
    out = make_run_dir(out_root, "e1_winrate_vs_random")
    rec = SelfPlayRecorder(out, config={
        "experiment": "e1_winrate_vs_random",
        "uct_c": 1.4,
        "sims_per_move_grid": list(sims_per_move_grid),
        "num_games": num_games,
    })

    game = CatanGame()
    for sims in sims_per_move_grid:
        for i in tqdm(range(num_games), desc=f"sims={sims}", leave=False):
            seed = seed_base + sims * 1_000 + i
            chance_rng = random.Random(seed)
            mcts_bot = _build_mcts_bot(game, sims=sims, seed=seed)
            bots = {0: mcts_bot, 1: _RandomBot(seed+1), 2: _RandomBot(seed+2), 3: _RandomBot(seed+3)}
            with rec.game(seed=seed) as g_rec:
                outcome = play_one_game(
                    game=game, bots=bots, seed=seed, chance_rng=chance_rng,
                    recorded_player=0, recorder_game=g_rec, mcts_bot=mcts_bot,
                )
                g_rec.finalize(
                    winner=outcome.winner,
                    final_vp=outcome.final_vp,
                    length_in_moves=outcome.length_in_moves,
                )
    rec.flush()
    return out


def cli_main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", type=Path, default=Path("mcts_study/runs"))
    p.add_argument("--num-games", type=int, default=50)
    p.add_argument("--sims-grid", type=int, nargs="+", default=[5, 25, 100, 400])
    p.add_argument("--seed-base", type=int, default=1_000_000)
    args = p.parse_args()
    out = main(
        out_root=args.out_root, num_games=args.num_games,
        sims_per_move_grid=args.sims_grid, seed_base=args.seed_base,
    )
    print(f"e1 wrote to {out}")


if __name__ == "__main__":
    cli_main()
