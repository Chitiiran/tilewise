"""Experiment 2: MCTS UCB exploration constant `c` sweep at fixed simulation budget."""
from __future__ import annotations

import argparse
import random
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


def main(
    *,
    out_root: Path,
    num_games: int = 50,
    c_grid: list[float] = (0.5, 1.0, 1.4, 2.0, 4.0),
    sims: int = 25,
    seed_base: int = 2_000_000,
    max_seconds: float = 300.0,
    resume: bool = True,
) -> Path:
    """v2: per-game cap, per-c-cell checkpoint, resume support."""
    out = make_run_dir(out_root, "e2_ucb_c_sweep")
    rec = SelfPlayRecorder(out, config={
        "experiment": "e2_ucb_c_sweep",
        "c_grid": list(c_grid),
        "sims_per_move": sims,
        "num_games": num_games,
        "max_seconds": max_seconds,
    })
    done = rec.done_seeds() if resume else set()

    game = CatanGame()
    for c in c_grid:
        for i in tqdm(range(num_games), desc=f"c={c}", leave=False):
            seed = seed_base + int(c * 100) * 1_000 + i
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
        rec.checkpoint(f"c={c}")
    rec.flush()
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
    args = p.parse_args()
    out = main(
        out_root=args.out_root, num_games=args.num_games,
        c_grid=args.c_grid, sims=args.sims, seed_base=args.seed_base,
        max_seconds=args.max_seconds, resume=not args.no_resume,
    )
    print(f"e2 wrote to {out}")


if __name__ == "__main__":
    cli_main()
