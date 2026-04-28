"""Tests for RustRolloutEvaluator.

The interesting comparison is against OpenSpiel's RandomRolloutEvaluator: both
should produce ~similar evaluation values (averages of random-rollout terminal
returns), but RustRolloutEvaluator is much faster because the rollout is one
PyO3 call instead of thousands.
"""
from __future__ import annotations

import time

import numpy as np
import pyspiel
from open_spiel.python.algorithms import mcts as os_mcts

from catan_mcts.adapter import CatanGame
from catan_mcts.evaluator import RustRolloutEvaluator


def _drive_to_player_decision(state):
    """Advance state past any leading chance nodes so evaluator gets a player turn."""
    while state.is_chance_node():
        outcomes = state.chance_outcomes()
        state.apply_action(int(outcomes[0][0]))


def test_returns_shape_matches_player_count():
    game = CatanGame()
    state = game.new_initial_state(seed=42)
    _drive_to_player_decision(state)

    evaluator = RustRolloutEvaluator(n_rollouts=1, base_seed=0)
    result = evaluator.evaluate(state)
    assert result.shape == (4,)
    # Single rollout: result is exactly the terminal returns of that rollout.
    plus = int(np.sum(result == 1.0))
    minus = int(np.sum(result == -1.0))
    assert plus == 1, f"expected 1 winner, got {result}"
    assert minus == 3, f"expected 3 losers, got {result}"


def test_does_not_mutate_input_state():
    game = CatanGame()
    state = game.new_initial_state(seed=42)
    _drive_to_player_decision(state)
    history_before = list(state._engine.action_history())

    evaluator = RustRolloutEvaluator(n_rollouts=2, base_seed=0)
    evaluator.evaluate(state)

    history_after = list(state._engine.action_history())
    assert history_after == history_before, \
        "evaluator mutated input state — the rollout must operate on a clone"


def test_n_rollouts_averages_results():
    """With n_rollouts=10 and varied seeds, the per-call return is a mean over
    10 rollouts. Each individual rollout sums to either -2 (terminal: one winner
    +1, three losers -1) or 0 (safety cap hit: [0,0,0,0]). So the mean sum is
    in [-2, 0]. Each player's average score is in [-1, 1]."""
    game = CatanGame()
    state = game.new_initial_state(seed=42)
    _drive_to_player_decision(state)

    evaluator = RustRolloutEvaluator(n_rollouts=10, base_seed=42)
    result = evaluator.evaluate(state)
    s = float(result.sum())
    assert -2.0 - 1e-6 <= s <= 0.0 + 1e-6, \
        f"mean rollout sum must be in [-2, 0]. got sum={s}, result={result}"
    for r in result:
        assert -1.0 <= float(r) <= 1.0


def test_prior_returns_uniform_over_legal_actions():
    game = CatanGame()
    state = game.new_initial_state(seed=42)
    _drive_to_player_decision(state)
    legal = state.legal_actions(state.current_player())

    evaluator = RustRolloutEvaluator()
    prior = evaluator.prior(state)
    assert len(prior) == len(legal)
    s = sum(p for _, p in prior)
    assert abs(s - 1.0) < 1e-9
    # All entries are legal action ids
    legal_set = set(int(a) for a in legal)
    for action, _ in prior:
        assert int(action) in legal_set


def test_prior_returns_chance_outcomes_at_chance_node():
    game = CatanGame()
    state = game.new_initial_state(seed=42)
    # Drive forward until we hit a chance node (Setup ends, Roll begins).
    for _ in range(2000):
        if state.is_chance_node():
            break
        state.apply_action(state.legal_actions()[0])
    assert state.is_chance_node()

    evaluator = RustRolloutEvaluator()
    prior = evaluator.prior(state)
    expected = state.chance_outcomes()
    assert prior == expected


def test_rust_evaluator_is_faster_than_python_random_rollout():
    """The whole point of this evaluator. If the speedup is < 5x we've broken
    something — the boundary-cost analysis predicts >>50x for 12k-step games."""
    game = CatanGame()
    state = game.new_initial_state(seed=42)
    _drive_to_player_decision(state)

    n_rollouts = 5

    # Python evaluator
    py_eval = os_mcts.RandomRolloutEvaluator(
        n_rollouts=n_rollouts, random_state=np.random.default_rng(seed=0)
    )
    t0 = time.perf_counter()
    py_result = py_eval.evaluate(state)
    py_time = time.perf_counter() - t0

    # Rust evaluator
    rust_eval = RustRolloutEvaluator(n_rollouts=n_rollouts, base_seed=0)
    t0 = time.perf_counter()
    rust_result = rust_eval.evaluate(state)
    rust_time = time.perf_counter() - t0

    speedup = py_time / max(rust_time, 1e-9)
    print(f"  python rollout: {py_time*1000:.1f} ms ({py_result})")
    print(f"  rust rollout:   {rust_time*1000:.1f} ms ({rust_result})")
    print(f"  speedup:        {speedup:.0f}x")
    # Each rollout sums to -2 (terminal) or 0 (safety cap); mean in [-2, 0].
    for r in (py_result, rust_result):
        s = float(r.sum())
        assert -2.0 - 1e-6 <= s <= 0.0 + 1e-6, f"rollout sum out of envelope: {s}"
    # The actual ratio depends on machine; expect at least 5x in release mode.
    assert speedup >= 5.0, f"expected at least 5x speedup, got {speedup:.1f}x"
