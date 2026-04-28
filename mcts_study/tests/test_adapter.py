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
