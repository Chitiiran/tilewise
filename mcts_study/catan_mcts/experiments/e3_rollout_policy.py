"""Experiment 3: MCTS rollout policy comparison — random vs heuristic, across sim budgets."""
from __future__ import annotations

import argparse
import random
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


def main(
    *,
    out_root: Path,
    num_games: int = 50,
    sims_grid: list[int] = (5, 25, 100),
    seed_base: int = 3_000_000,
    max_seconds: float = 300.0,
    resume: bool = True,
) -> Path:
    """v2: per-game cap, per-(policy,sims)-cell checkpoint, resume support."""
    out = make_run_dir(out_root, "e3_rollout_policy")
    rec = SelfPlayRecorder(out, config={
        "experiment": "e3_rollout_policy",
        "uct_c": 1.4,
        "sims_grid": list(sims_grid),
        "num_games": num_games,
        "rollout_policies": ["random", "heuristic"],
        "max_seconds": max_seconds,
    })
    done = rec.done_seeds() if resume else set()

    game = CatanGame()
    for policy_name in ("random", "heuristic"):
        for sims in sims_grid:
            for i in tqdm(range(num_games), desc=f"{policy_name} sims={sims}", leave=False):
                seed = seed_base + (0 if policy_name == "random" else 1) * 1_000_000 + sims * 1_000 + i
                if seed in done:
                    continue
                chance_rng = random.Random(seed)
                rng = np.random.default_rng(seed)
                if policy_name == "random":
                    # Rust-side rollout — see RustRolloutEvaluator docstring.
                    evaluator = RustRolloutEvaluator(n_rollouts=1, base_seed=seed)
                else:
                    # Heuristic policy is intentionally Python-side: it's the
                    # comparison baseline against random rollouts. Speed parity
                    # would mask the very thing we're measuring.
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
            rec.checkpoint(f"{policy_name}-sims={sims}")
    rec.flush()
    return out


def cli_main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", type=Path, default=Path("runs"))
    p.add_argument("--num-games", type=int, default=50)
    p.add_argument("--sims-grid", type=int, nargs="+", default=[5, 25, 100])
    p.add_argument("--seed-base", type=int, default=3_000_000)
    p.add_argument("--max-seconds", type=float, default=300.0)
    p.add_argument("--no-resume", action="store_true")
    args = p.parse_args()
    out = main(
        out_root=args.out_root, num_games=args.num_games,
        sims_grid=args.sims_grid, seed_base=args.seed_base,
        max_seconds=args.max_seconds, resume=not args.no_resume,
    )
    print(f"e3 wrote to {out}")


if __name__ == "__main__":
    cli_main()
