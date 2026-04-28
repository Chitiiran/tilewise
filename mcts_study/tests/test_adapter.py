import pytest
from catan_mcts.adapter import CatanGame
from catan_mcts import ACTION_SPACE_SIZE


def test_game_construction():
    game = CatanGame()
    assert game.num_distinct_actions() == ACTION_SPACE_SIZE
    assert game.num_players() == 4


def test_initial_state_is_not_terminal():
    game = CatanGame()
    state = game.new_initial_state()
    assert not state.is_terminal()


def test_initial_state_has_legal_actions():
    game = CatanGame()
    state = game.new_initial_state()
    legal = state.legal_actions()
    assert len(legal) > 0
    assert all(0 <= a < ACTION_SPACE_SIZE for a in legal)


import pyspiel
from catan_mcts.adapter import CatanGame


def test_chance_player_id_matches_pyspiel():
    # OpenSpiel uses pyspiel.PlayerId.CHANCE = -1.
    game = CatanGame()
    state = game.new_initial_state(seed=42)
    # Drive forward until a chance node fires (Roll phase).
    for _ in range(2000):
        if state.is_chance_node():
            assert state.current_player() == pyspiel.PlayerId.CHANCE
            outcomes = state.chance_outcomes()
            assert len(outcomes) > 0
            total = sum(p for _, p in outcomes)
            assert abs(total - 1.0) < 1e-9
            return
        state.apply_action(state.legal_actions()[0])
    raise AssertionError("no chance node reached in 2000 steps")


def test_chance_node_drives_dice():
    game = CatanGame()
    state = game.new_initial_state(seed=42)
    for _ in range(2000):
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            value, _ = outcomes[0]
            state.apply_action(int(value))
            # Either we're past chance now, or we entered another chance node.
            assert not state.is_chance_node() or state.current_player() == pyspiel.PlayerId.CHANCE
            return
        state.apply_action(state.legal_actions()[0])
    raise AssertionError("no chance node reached")
