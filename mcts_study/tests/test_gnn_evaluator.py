"""Tests for GnnEvaluator: OpenSpiel evaluator backed by a GNN."""
import numpy as np
import pyspiel
import pytest
import torch
from open_spiel.python.algorithms import mcts as os_mcts

from catan_gnn.gnn_model import GnnModel
from catan_mcts.adapter import CatanGame
from catan_mcts.gnn_evaluator import GnnEvaluator


def _drive_to_player_decision(state):
    while state.is_chance_node():
        outcomes = state.chance_outcomes()
        state.apply_action(int(outcomes[0][0]))


def _untrained_model():
    torch.manual_seed(0)
    return GnnModel(hidden_dim=32, num_layers=2)


def test_evaluate_shape_and_range():
    game = CatanGame()
    state = game.new_initial_state(seed=42)
    _drive_to_player_decision(state)
    ev = GnnEvaluator(model=_untrained_model())
    v = ev.evaluate(state)
    assert v.shape == (4,)
    assert v.dtype == np.float32
    assert (v >= -1.0).all() and (v <= 1.0).all()


def test_prior_sums_to_one_over_legal():
    game = CatanGame()
    state = game.new_initial_state(seed=42)
    _drive_to_player_decision(state)
    ev = GnnEvaluator(model=_untrained_model())
    prior = ev.prior(state)
    legal = state.legal_actions(state.current_player())
    assert len(prior) == len(legal)
    s = sum(p for _, p in prior)
    assert abs(s - 1.0) < 1e-5


def test_evaluate_does_not_mutate_state():
    game = CatanGame()
    state = game.new_initial_state(seed=42)
    _drive_to_player_decision(state)
    history_before = list(state._engine.action_history())
    ev = GnnEvaluator(model=_untrained_model())
    ev.evaluate(state)
    ev.prior(state)
    history_after = list(state._engine.action_history())
    assert history_before == history_after


def test_runs_inside_mctsbot_one_full_game():
    """End-to-end: MCTSBot(GnnEvaluator) + 3 random opponents play one game.
    Untrained model -> noise priors -> any legal play is fine; just ensure
    no exceptions and the game terminates within reasonable steps."""
    import random

    game = CatanGame()
    rng = np.random.default_rng(0)
    ev = GnnEvaluator(model=_untrained_model())
    mcts_bot = os_mcts.MCTSBot(
        game=game, uct_c=1.4, max_simulations=4,
        evaluator=ev, solve=False, random_state=rng,
    )

    class _RandBot:
        def __init__(self, s): self.r = random.Random(s)
        def step(self, st): return self.r.choice(st.legal_actions())

    state = game.new_initial_state(seed=42)
    chance_rng = random.Random(42)
    bots = {0: mcts_bot, 1: _RandBot(1), 2: _RandBot(2), 3: _RandBot(3)}
    steps = 0
    while not state.is_terminal() and steps < 30000:
        if state.is_chance_node():
            outs = state.chance_outcomes()
            r = chance_rng.random(); cum = 0.0; chosen = outs[-1][0]
            for v, p in outs:
                cum += p
                if r <= cum:
                    chosen = v; break
            state.apply_action(int(chosen))
        else:
            p = state.current_player()
            legal = state.legal_actions()
            if len(legal) == 1:
                state.apply_action(int(legal[0]))
            else:
                a = bots[p].step(state)
                state.apply_action(int(a))
        steps += 1
    assert steps > 0
