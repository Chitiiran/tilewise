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


def test_board_payload_present():
    sess = GameSession(_setup())
    board = sess.board_payload()
    assert "layout" in board and "png_b64" in board
    assert board["png_b64"]
