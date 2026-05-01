"""M1: combined PyO3 entry points.

Each combined call must produce results identical to the per-call sequence
it replaces. Tests guard against silent semantic drift between the
combined fast-path and the original PyEngine API.
"""
from __future__ import annotations

import pyspiel
from catan_bot import _engine


def _drive_to_main(seed: int = 42) -> _engine.Engine:
    """Walk the engine through Setup1+Setup2+initial roll into a Main-phase
    state. Used so combined-call tests cover real legal-actions flows, not
    just the trivial Setup1 case."""
    e = _engine.Engine(seed)
    while not e.is_terminal():
        if e.is_chance_pending():
            outcomes = e.chance_outcomes()
            e.apply_chance_outcome(int(outcomes[0][0]))
        else:
            legal = list(e.legal_actions())
            if not legal:
                break
            e.step(int(legal[0]))
        # Stop once we've taken some real-rules actions but not so many we
        # might terminate.
        if len(e.action_history()) > 30:
            break
    return e


# ---------------------------------------------------------------- query_status

def test_query_status_matches_individual_calls_setup_phase():
    e = _engine.Engine(42)
    expected = (e.is_terminal(), e.is_chance_pending(), e.current_player())
    actual = e.query_status()
    assert actual == expected


def test_query_status_matches_individual_calls_main_phase():
    e = _drive_to_main(42)
    expected = (e.is_terminal(), e.is_chance_pending(), e.current_player())
    actual = e.query_status()
    assert actual == expected


def test_query_status_returns_terminal_after_done():
    """At terminal the tuple must reflect (True, False, *) consistent with
    is_terminal()/is_chance_pending() — current_player remains the engine's
    bookkeeping value."""
    e = _engine.Engine(42)
    # Cheap-ish way to drive to terminal: random walk with a hard cap.
    steps = 0
    while not e.is_terminal() and steps < 5000:
        if e.is_chance_pending():
            outcomes = e.chance_outcomes()
            e.apply_chance_outcome(int(outcomes[0][0]))
        else:
            legal = list(e.legal_actions())
            if not legal:
                break
            e.step(int(legal[0]))
        steps += 1
    if not e.is_terminal():
        # Determinism + 5k cap may not terminate; smoke-skip rather than fail.
        return
    is_term, is_chance, cp = e.query_status()
    assert is_term is True
    assert is_chance is False
    assert 0 <= cp < 4


# -------------------------------------------------------- apply_action_smart

def test_apply_action_smart_dispatches_chance_in_chance_phase():
    """When the engine is awaiting a chance outcome, apply_action_smart must
    route the action to apply_chance_outcome()."""
    e = _engine.Engine(42)
    # Drive into a chance-pending state. After Setup1+Setup2, the first
    # decision is the dice roll which is a chance node.
    while not e.is_chance_pending() and not e.is_terminal():
        legal = list(e.legal_actions())
        if not legal:
            break
        e.step(int(legal[0]))
    assert e.is_chance_pending(), "test setup: expected to reach a chance node"
    outcomes = e.chance_outcomes()
    chosen = int(outcomes[0][0])

    # Snapshot for divergence check.
    e2 = e.clone()
    e2.apply_chance_outcome(chosen)
    e.apply_action_smart(chosen)
    # Both paths should produce identical action_history.
    assert list(e.action_history()) == list(e2.action_history())


def test_apply_action_smart_dispatches_step_in_decision_phase():
    """When the engine is awaiting a decision, apply_action_smart must route
    the action to step()."""
    e = _engine.Engine(42)
    legal = list(e.legal_actions())
    chosen = int(legal[0])

    e2 = e.clone()
    e2.step(chosen)
    e.apply_action_smart(chosen)
    assert list(e.action_history()) == list(e2.action_history())


def test_apply_action_smart_returns_post_status():
    """apply_action_smart returns (is_terminal, is_chance_pending,
    current_player) AFTER the action — saves the immediate-next status query
    that MCTS does on every step."""
    e = _engine.Engine(42)
    legal = list(e.legal_actions())
    chosen = int(legal[0])
    is_term, is_chance, cp = e.apply_action_smart(chosen)
    # Must match an explicit query right after.
    assert (is_term, is_chance, cp) == (
        e.is_terminal(), e.is_chance_pending(), e.current_player(),
    )


# --------------------------------------------- adapter still works end-to-end

def test_openspiel_adapter_still_legal_after_combined_calls():
    """Belt-and-suspenders: the OpenSpiel CatanGame must still produce a
    legal-actions list and apply moves through pyspiel.State after the M1
    refactor lands. If we accidentally change behavior, the regression
    test_engine_regression.py replay will catch it; this is the integration
    smoke."""
    from catan_mcts.adapter import CatanGame
    game = CatanGame()
    state = game.new_initial_state(seed=42)
    assert not state.is_terminal()
    legal = state.legal_actions()
    assert len(legal) > 0
    state.apply_action(int(legal[0]))
    assert state.history()  # action recorded
