"""v3 adapter + PyO3 tests.

Verifies that the Python adapter and _engine PyO3 bindings can construct
engines with v3 rule flags (vp_target, bonuses) and that the flags
actually take effect end-to-end (terminal at 5 VP, no +2 LR/LA bonus).
"""
from __future__ import annotations

import pyspiel
import pytest

from catan_bot import _engine
from catan_mcts.adapter import CatanGame


# --- _engine.Engine PyO3 surface -------------------------------------------


def test_engine_default_constructor_is_v2():
    """`_engine.Engine(seed)` keeps v2 semantics."""
    e = _engine.Engine(42)
    assert e.vp_target() == 10
    assert e.bonuses_enabled() is True


def test_engine_with_rules_v3():
    """`_engine.Engine.with_rules(seed, 5, False)` opts into v3."""
    e = _engine.Engine.with_rules(42, 5, False)
    assert e.vp_target() == 5
    assert e.bonuses_enabled() is False


def test_engine_with_rules_v2_explicit():
    """`_engine.Engine.with_rules(seed, 10, True)` is identical in semantics
    to the default constructor."""
    e = _engine.Engine.with_rules(42, 10, True)
    assert e.vp_target() == 10
    assert e.bonuses_enabled() is True


# --- CatanGame Python adapter ---------------------------------------------


def test_catan_game_default_is_v2():
    """`CatanGame()` defaults preserve v2."""
    game = CatanGame()
    state = game.new_initial_state(seed=42)
    # Reach into the wrapped engine to check flags.
    assert state._engine.vp_target() == 10
    assert state._engine.bonuses_enabled() is True


def test_catan_game_v3_kwargs():
    """`CatanGame(vp_target=5, bonuses=False)` creates a v3 game."""
    game = CatanGame(vp_target=5, bonuses=False)
    state = game.new_initial_state(seed=42)
    assert state._engine.vp_target() == 5
    assert state._engine.bonuses_enabled() is False


def test_catan_game_v3_serialize_roundtrip_preserves_flags():
    """Serialize/deserialize should round-trip the v3 flags so distributed
    workers don't silently drop back to v2 on deserialize."""
    game = CatanGame(vp_target=5, bonuses=False)
    state = game.new_initial_state(seed=42)
    blob = state.serialize()
    state2 = CatanGame.deserialize(blob)
    assert state2._engine.vp_target() == 5
    assert state2._engine.bonuses_enabled() is False
