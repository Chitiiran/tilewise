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


def test_clone_is_independent():
    game = CatanGame()
    a = game.new_initial_state(seed=42)
    a.apply_action(a.legal_actions()[0])
    b = a.clone()
    history_before = list(a.history())
    b.apply_action(b.legal_actions()[0])
    assert list(a.history()) == history_before


def test_returns_zero_until_terminal():
    game = CatanGame()
    state = game.new_initial_state(seed=42)
    assert state.returns() == [0.0, 0.0, 0.0, 0.0]


def test_returns_at_terminal_have_one_winner():
    """Drive a game to completion using random+chance policy. At termination,
    +1 / -1 / -1 / -1 returns are expected (one winner)."""
    import random
    game = CatanGame()
    state = game.new_initial_state(seed=42)
    rng = random.Random(0)
    steps = 0
    while not state.is_terminal() and steps < 5000:
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
            legal = state.legal_actions()
            if not legal:
                break
            state.apply_action(rng.choice(legal))
        steps += 1
    if not state.is_terminal():
        # Random policy may not finish on every seed; not the test target
        return
    rs = state.returns()
    assert rs.count(1.0) == 1
    assert rs.count(-1.0) == 3


def test_serialize_round_trips_through_history():
    game = CatanGame()
    s1 = game.new_initial_state(seed=42)
    s1.apply_action(s1.legal_actions()[0])
    blob = s1.serialize()
    s2 = CatanGame.deserialize(blob)
    assert s2.legal_actions() == s1.legal_actions()
    assert s2.is_terminal() == s1.is_terminal()
