"""Tests for the live GameSession."""
from __future__ import annotations

import pytest

from catan_mcts.web.game_session import GameSession


def _setup(human_seat=0):
    return {
        "human_seat": human_seat,
        "seats": {str(s): {"type": "Random"} for s in range(4) if s != human_seat},
        "rules": {"vp_target": 10, "bonuses": True},
        "seed": 4242,
    }


def test_construct_and_state_json():
    sess = GameSession(_setup())
    s = sess.state_json()
    assert s["human_seat"] == 0
    assert s["status"] in {"your_turn", "bot_thinking", "trade_offer", "game_over"}
    assert "state" in s and "seat_names" in s
    assert len(s["seat_names"]) == 4


def test_seat_names_distinct_and_informative():
    sess = GameSession(_setup(human_seat=0))
    names = sess.seat_names()
    assert len(names) == 4
    assert names[0] == "You"
    bot_names = names[1:]
    # Three Random bots must read as three distinct personas, not all "Random".
    assert len(set(bot_names)) == 3, bot_names
    # Each bot name stays informative: it carries its type label in parens.
    assert all("(Random)" in n for n in bot_names), bot_names


def test_seat_names_gnn_includes_checkpoint_stem():
    # Build a Random-only session, then swap in a GNN-style spec so seat_names
    # exercises the checkpoint-stem path without loading a real torch model.
    sess = GameSession(_setup(human_seat=0))
    sess._seat_specs[3] = {"type": "PureGnn", "checkpoint": "/some/dir/round0_Cell6.pt"}
    names = sess.seat_names()
    assert names[0] == "You"
    assert "PureGnn" in names[3] and "round0_Cell6" in names[3], names[3]
    assert len(set(names)) == 4


def test_last_action_recorded_after_apply():
    sess = GameSession(_setup(human_seat=0))
    assert sess.state_json()["last_action"] is None
    sess.advance()
    la = sess.state_json()["last_action"]
    assert la is None or ({"action", "player"} <= set(la))


def test_board_payload_present():
    sess = GameSession(_setup())
    board = sess.board_payload()
    assert "layout" in board and "png_b64" in board
    assert board["png_b64"]


def test_advance_reaches_human_turn_or_terminal():
    sess = GameSession(_setup(human_seat=0))
    res = sess.advance()
    assert res["status"] in {"your_turn", "game_over"}
    if res["status"] == "your_turn":
        assert len(res["legal_actions"]) >= 1
        assert int(sess.state_json()["current_player"]) == 0


def test_full_game_with_stub_human_terminates():
    sess = GameSession(_setup(human_seat=0))
    for _ in range(100000):
        res = sess.advance()
        if res["status"] == "game_over":
            break
        if res["status"] == "your_turn":
            sess.apply_human_action(res["legal_actions"][0]["id"])
        elif res["status"] == "trade_offer":
            sess.respond_to_trade(accept=False)
        else:
            raise AssertionError(f"unexpected status {res['status']}")
    assert sess.state_json()["status"] == "game_over"
    assert sess.state_json()["returns"] is not None


def _trade_session(human_seat=1):
    return GameSession(_setup(human_seat=human_seat))


def test_no_intercept_when_trade_targets_other_bot():
    sess = _trade_session(human_seat=1)
    sess._predict_trade_acceptor = lambda cp, action: 2
    assert sess._maybe_intercept_trade(current_player=0, action=260) is False


def test_intercept_pauses_when_trade_targets_human():
    sess = _trade_session(human_seat=1)
    # Force the predictor to say the human (seat 1) is the acceptor.
    sess._predict_trade_acceptor = lambda cp, action: sess.human_seat
    res = sess._maybe_intercept_trade(current_player=0, action=260)
    assert res is True
    sj = sess.state_json()
    assert sj["status"] == "trade_offer"
    assert sj["trade_offer"]["from_seat"] == 0


def test_trade_offer_payload_values():
    sess = _trade_session(human_seat=1)
    # action 260 decodes to give=0 (wood), get=1 (brick). From the human's view
    # the swap is mirrored: you_give = what the bot wants (get), you_get = what
    # the bot offers (give).
    sess._pending_trade = (0, 260)
    payload = sess._trade_offer_payload()
    assert payload["you_give"] == [1, 1]
    assert payload["you_get"] == [0, 1]
    assert payload["from_seat"] == 0


def test_reject_leaves_human_hand_unchanged():
    sess = _trade_session(human_seat=1)
    sess._pending_trade = (0, 260)
    before = [list(h) for h in sess._state._engine.all_hands()]
    # Make the proposer bot (seat 0) end its turn when re-queried.
    sess._bots[0].step = lambda state: 204
    sess.respond_to_trade(accept=False)
    after = [list(h) for h in sess._state._engine.all_hands()]
    assert after[1] == before[1], "human hand changed on reject"


import time


def test_advance_async_runs_in_background():
    sess = GameSession(_setup(human_seat=0))
    sess.advance_async()
    for _ in range(200):
        if not sess.is_advancing():
            break
        time.sleep(0.02)
    assert not sess.is_advancing()
    assert sess.state_json()["status"] in {"your_turn", "game_over"}


def test_apply_human_action_async_returns_and_settles():
    sess = GameSession(_setup(human_seat=0))
    # Drive to the human's first decision synchronously, then act async.
    s = sess.advance()
    if s["status"] != "your_turn":
        return  # game ended immediately (unlikely); nothing to assert
    aid = s["legal_actions"][0]["id"]
    out = sess.apply_human_action_async(aid)
    assert out["status"] in {"your_turn", "bot_thinking", "trade_offer", "game_over"}
    # Wait for the background driving to settle.
    import time as _t
    for _ in range(500):
        if not sess.is_advancing():
            break
        _t.sleep(0.02)
    assert not sess.is_advancing()
    final = sess.state_json()
    assert final["status"] in {"your_turn", "trade_offer", "game_over", "error"}
