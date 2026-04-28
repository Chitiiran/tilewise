"""End-to-end smoke: OpenSpiel MCTSBot plays a full game vs 3 random opponents.

Purpose: prove the OpenSpiel <-> catan_bot adapter integration works end-to-end —
MCTSBot accepts our CatanGame, runs simulations, and the search loop drives a full
game to terminal. NOT a strength measurement.

Calibration: in release mode, real games are ~12-15k steps because Phase-0 made
dice + steal honest chance points. With max_simulations=5 and rollouts going to
terminal, one MCTS move costs ~5 * 12k = 60k state ops; ~1000 MCTS-decisions per
game → ~6e7 ops total, under a minute. Don't crank max_simulations here — that's
what production runs (Plan 3) are for.
"""
import random

import numpy as np
import pyspiel
from open_spiel.python.algorithms import mcts

from catan_mcts.adapter import CatanGame


class _RandomBotShim:
    """Tiny random bot — picks uniformly from legal actions."""

    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def step(self, state) -> int:
        return self._rng.choice(state.legal_actions())


def test_mcts_vs_random_full_game():
    game = CatanGame()
    state = game.new_initial_state(seed=42)

    rng = np.random.default_rng(seed=42)
    rand_eval = mcts.RandomRolloutEvaluator(n_rollouts=1, random_state=rng)
    mcts_bot = mcts.MCTSBot(
        game=game,
        uct_c=1.4,
        max_simulations=5,  # smoke only; production runs use 50-4000
        evaluator=rand_eval,
        solve=False,
        random_state=rng,
    )
    bots = {0: mcts_bot, 1: _RandomBotShim(1), 2: _RandomBotShim(2), 3: _RandomBotShim(3)}

    steps = 0
    # Real Tier-1 games take ~12-15k steps under random/greedy policies because each
    # turn includes a Roll chance node + frequent discards. 30k is a generous cap.
    while not state.is_terminal() and steps < 30000:
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            r = rng.random()
            cum = 0.0
            chosen = outcomes[-1][0]
            for v, p in outcomes:
                cum += p
                if r <= cum:
                    chosen = v
                    break
            state.apply_action(int(chosen))
        else:
            p = state.current_player()
            bot = bots[p]
            action = bot.step(state)
            state.apply_action(int(action))
        steps += 1

    assert state.is_terminal(), f"game did not terminate in 30000 steps (got {steps})"
    rs = state.returns()
    assert rs.count(1.0) == 1
    assert rs.count(-1.0) == 3
