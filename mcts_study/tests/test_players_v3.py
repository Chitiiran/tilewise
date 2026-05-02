"""v3.4 tests — LookaheadMctsV3 player + VP-aware sim schedule."""
from __future__ import annotations

import pytest

from catan_mcts.adapter import CatanGame
from catan_mcts.players_v3 import (
    SIM_BASE,
    SIM_DECAY,
    SIM_FLOOR,
    _AdaptiveSimsMCTSBot,
    build_lookahead_mcts_v3,
    sims_for_player,
)


# --- Schedule formula -------------------------------------------------------


@pytest.mark.parametrize("vp,expected", [
    (0, 200),
    (1, 140),
    (2, 98),
    (3, 69),
    (4, 50),
    (5, 50),  # floor
    (6, 50),  # floor (defensive)
])
def test_sims_for_player_canonical_table(vp: int, expected: int):
    """Schedule must match the exact spec §5 numbers."""
    assert sims_for_player(vp) == expected


def test_sims_for_player_monotone_decreasing():
    """Schedule must be non-increasing in VP."""
    prev = sims_for_player(0)
    for vp in range(1, 10):
        cur = sims_for_player(vp)
        assert cur <= prev, f"VP={vp}: {cur} > {prev}"
        prev = cur


def test_sims_for_player_floor_holds():
    for vp in range(0, 20):
        assert sims_for_player(vp) >= SIM_FLOOR


def test_constants_consistent():
    # Defensive — if anyone tweaks the constants, the canonical table test
    # above will catch it; this test pins the formula's algebra.
    assert sims_for_player(0) == SIM_BASE
    assert SIM_FLOOR < SIM_BASE
    assert 0 < SIM_DECAY < 1


# --- Player construction ----------------------------------------------------


def test_build_lookahead_mcts_v3_runs_one_step():
    """Smoke: construct a v3 game + v3 player and run one step."""
    game = CatanGame(vp_target=5, bonuses=False)
    state = game.new_initial_state(seed=42)
    bot = build_lookahead_mcts_v3(game, lookahead_depth=5, seed=42, base_sims=50)
    # Setup1 places a settlement first; the bot returns an action.
    action = bot.step(state)
    assert isinstance(action, int)
    assert action in state.legal_actions(state.current_player())


def test_build_lookahead_mcts_v3_rejects_low_base_sims():
    game = CatanGame(vp_target=5, bonuses=False)
    with pytest.raises(ValueError, match="below SIM_FLOOR"):
        build_lookahead_mcts_v3(game, base_sims=10)


def test_step_sets_max_simulations_from_acting_vp_at_zero():
    """At the start of a game everyone has 0 VP, so the bot's step() should
    leave max_simulations at SIM_BASE = 200."""
    game = CatanGame(vp_target=5, bonuses=False)
    state = game.new_initial_state(seed=42)
    bot = build_lookahead_mcts_v3(game, lookahead_depth=2, seed=42, base_sims=200)
    bot.step(state)
    # After step(), the bot should have set max_simulations based on
    # the acting player's VP (0) — exactly SIM_BASE.
    assert bot.max_simulations == 200


def test_step_uses_floor_for_high_vp():
    """If we manually set the engine's VP to 5+, max_simulations should drop
    to the floor."""
    game = CatanGame(vp_target=5, bonuses=False)
    state = game.new_initial_state(seed=42)
    bot = build_lookahead_mcts_v3(game, lookahead_depth=2, seed=42, base_sims=200)
    # We can't mutate VP on the engine from Python directly — but we can
    # verify the FORMULA via a synthetic state-less call.
    assert sims_for_player(5) == SIM_FLOOR
    assert sims_for_player(8) == SIM_FLOOR


def test_full_game_runs_with_v3_player():
    """A v3 self-play game with all-LookaheadMctsV3 players reaches a
    terminal state within a reasonable number of moves and ends at <= 5 VP
    for every player."""
    game = CatanGame(vp_target=5, bonuses=False)
    state = game.new_initial_state(seed=42)
    # base_sims=50 keeps this test fast.
    bots = [build_lookahead_mcts_v3(game, lookahead_depth=3, seed=42 + i, base_sims=50)
            for i in range(4)]
    max_moves = 2000
    moves = 0
    while not state.is_terminal() and moves < max_moves:
        cp = state.current_player()
        if cp < 0:
            # Chance node — sample uniformly by weight.
            outcomes = state.chance_outcomes()
            assert outcomes
            import numpy as np
            ids = [int(a) for a, _ in outcomes]
            probs = np.array([p for _, p in outcomes], dtype=np.float64)
            probs = probs / probs.sum()
            chosen = int(np.random.default_rng(42 + moves).choice(ids, p=probs))
            state.apply_action(chosen)
        else:
            action = bots[cp].step(state)
            state.apply_action(action)
        moves += 1
    assert state.is_terminal(), f"game didn't terminate in {max_moves} moves"
    # Under v3 rules the game ends when someone hits exactly 5 VP and the
    # loser VPs cannot exceed 5 either (no LR/LA path to higher).
    for p in range(4):
        assert state._engine.vp(p) <= 5, f"P{p} ended with {state._engine.vp(p)} VP > 5 cap"
