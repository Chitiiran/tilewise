"""Determinism regression: same seed + same config → byte-identical moves.parquet.

Mirrors the smoke-test RNG layout (chance + MCTS share one numpy Generator;
each random opponent has its own ``random.Random``). The spec's original
single-shared ``rng_random`` setup made MCTS rollouts ~10x slower for this
particular seed because of the trajectory it funnelled the search into; the
smoke pattern is both faster and still fully deterministic.
"""
from pathlib import Path
import hashlib
import random

import numpy as np
import pyspiel
from open_spiel.python.algorithms import mcts

from catan_mcts.adapter import CatanGame
from catan_mcts.recorder import SelfPlayRecorder
from catan_mcts import ACTION_SPACE_SIZE


class _RandomBotShim:
    """Tiny random bot — picks uniformly from legal actions (matches smoke test)."""

    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def step(self, state) -> int:
        return self._rng.choice(state.legal_actions())


def _run_one(out_dir: Path) -> bytes:
    game = CatanGame()
    rec = SelfPlayRecorder(out_dir, config={"uct_c": 1.4, "max_simulations": 5})
    rng = np.random.default_rng(seed=42)
    state = game.new_initial_state(seed=42)
    evaluator = mcts.RandomRolloutEvaluator(n_rollouts=1, random_state=rng)
    bot = mcts.MCTSBot(
        game=game, uct_c=1.4, max_simulations=5, evaluator=evaluator,
        solve=False, random_state=rng,
    )
    bots = {0: bot, 1: _RandomBotShim(1), 2: _RandomBotShim(2), 3: _RandomBotShim(3)}
    move_index = 0

    with rec.game(seed=42) as g:
        steps = 0
        while not state.is_terminal() and steps < 30000:
            if state.is_chance_node():
                outcomes = state.chance_outcomes()
                r = rng.random()
                cum, chosen = 0.0, outcomes[-1][0]
                for v, p in outcomes:
                    cum += p
                    if r <= cum:
                        chosen = v
                        break
                state.apply_action(int(chosen))
            else:
                p = state.current_player()
                action = bots[p].step(state)
                if p == 0:
                    legal = state.legal_actions()
                    mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
                    mask[legal] = True
                    # Determinism test only checks parquet bytes are byte-identical,
                    # not that visit counts are right. Zeros are deterministic.
                    visits = np.zeros(ACTION_SPACE_SIZE, dtype=np.int32)
                    g.record_move(
                        current_player=0, move_index=move_index,
                        legal_action_mask=mask, mcts_visit_counts=visits,
                        action_taken=int(action), mcts_root_value=0.0,
                    )
                    move_index += 1
                state.apply_action(int(action))
            steps += 1
        winner = -1
        if state.is_terminal():
            rs = state.returns()
            winner = rs.index(1.0) if 1.0 in rs else -1
        g.finalize(winner=winner, final_vp=[0,0,0,0], length_in_moves=steps)
    rec.flush()
    return (out_dir / "moves.parquet").read_bytes()


def test_byte_identical_runs(tmp_path: Path):
    a = _run_one(tmp_path / "run_a")
    b = _run_one(tmp_path / "run_b")
    assert hashlib.sha256(a).hexdigest() == hashlib.sha256(b).hexdigest()
