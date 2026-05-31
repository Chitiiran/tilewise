"""Tests for bot discovery + construction."""
from __future__ import annotations

import pytest


def test_list_types_includes_core_bots():
    from catan_mcts.web import bot_registry
    types = {t["id"] for t in bot_registry.list_types()}
    assert {"Random", "Greedy", "LookaheadMctsV3", "PureGnn", "GnnMcts"} <= types


def test_build_random_bot():
    from catan_mcts.web import bot_registry
    from catan_mcts.adapter import CatanGame
    game = CatanGame()
    bot = bot_registry.build({"type": "Random"}, game=game, seed=7)
    state = game.new_initial_state(seed=7)
    action = bot.step(state)
    assert action in state.legal_actions()


def test_build_greedy_bot():
    from catan_mcts.web import bot_registry
    from catan_mcts.adapter import CatanGame
    game = CatanGame()
    bot = bot_registry.build({"type": "Greedy"}, game=game, seed=1)
    state = game.new_initial_state(seed=1)
    assert bot.step(state) in state.legal_actions()


def test_build_unknown_type_raises():
    from catan_mcts.web import bot_registry
    from catan_mcts.adapter import CatanGame
    with pytest.raises(ValueError, match="unknown bot type"):
        bot_registry.build({"type": "Nope"}, game=CatanGame(), seed=0)
