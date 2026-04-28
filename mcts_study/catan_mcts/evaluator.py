"""Custom OpenSpiel Evaluator that pushes the rollout into Rust.

Rationale: `mcts.RandomRolloutEvaluator` runs the rollout in a Python while-loop,
crossing the PyO3 boundary once per step (legal_actions, apply_action, etc).
Empirically that's ~33µs/step × ~12k steps per Tier-1 game = ~400ms per rollout.
At sims=5 with ~3000 MCTS-decision turns per recorded game, that's ~99 minutes
per recorded game.

`RustRolloutEvaluator` instead calls `engine.random_rollout_to_terminal(seed)`
once per simulation. That's a single PyO3 call wrapping a pure-Rust full-game
rollout (~28µs in release mode). Expected ~1000x speedup on the rollout phase
of MCTS, which is the bottleneck for recorded production runs.

The Rust rollout uses an independent SmallRng seeded by the caller — so the
search-state's engine RNG is unaffected. Each evaluator call uses a fresh seed
mixed from the seed-counter so multiple simulations from the same root produce
distinct trajectories.
"""
from __future__ import annotations

import numpy as np
from open_spiel.python.algorithms import mcts as os_mcts


class RustRolloutEvaluator(os_mcts.Evaluator):
    """Evaluator that runs the full rollout in Rust via `random_rollout_to_terminal`.

    Parameters
    ----------
    n_rollouts : int
        Number of independent rollouts averaged per `evaluate` call. Each
        rollout uses a different seed derived from the evaluator's counter.
    base_seed : int
        Base seed for rollout RNGs. Combined with a per-call counter to
        produce a fresh seed per rollout.
    """

    def __init__(self, n_rollouts: int = 1, base_seed: int = 0) -> None:
        self.n_rollouts = int(n_rollouts)
        self._base_seed = int(base_seed)
        self._counter = 0

    def evaluate(self, state):
        """Run `n_rollouts` Rust-side rollouts and average their returns."""
        result = np.zeros(state.num_players(), dtype=np.float32)
        for _ in range(self.n_rollouts):
            self._counter += 1
            rollout_seed = (self._base_seed * 1_000_003) ^ self._counter
            # Clone the OpenSpiel state to keep the search tree's state intact.
            working = state.clone()
            # Reach into our adapter's underlying engine and run the Rust rollout.
            # `working` is a CatanState — its `_engine` is the catan_bot._engine.Engine.
            returns = working._engine.random_rollout_to_terminal(rollout_seed & 0xFFFF_FFFF_FFFF_FFFF)
            result += np.asarray(returns, dtype=np.float32)
        return result / self.n_rollouts

    def prior(self, state):
        """Equal probability for all actions (matches RandomRolloutEvaluator)."""
        if state.is_chance_node():
            return state.chance_outcomes()
        legal = state.legal_actions(state.current_player())
        if not legal:
            return []
        p = 1.0 / len(legal)
        return [(int(a), p) for a in legal]
