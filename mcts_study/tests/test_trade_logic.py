"""Tests for engine-faithful trade-match prediction."""
from __future__ import annotations


def test_decode_propose_trade_give_get():
    from catan_mcts.web import trade_logic
    give, get = trade_logic.decode_propose_trade(260)
    assert give == 0 and get == 1


def test_first_acceptor_seat_order():
    from catan_mcts.web import trade_logic
    hands = [
        [1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0],
    ]
    acceptor = trade_logic.first_acceptor(current_player=0, give=0, get=1, hands=hands)
    assert acceptor == 2


def test_no_acceptor_returns_minus_one():
    from catan_mcts.web import trade_logic
    hands = [[1, 0, 0, 0, 0]] + [[0, 0, 0, 0, 0]] * 3
    assert trade_logic.first_acceptor(0, 0, 1, hands) == -1


def test_would_match_human():
    from catan_mcts.web import trade_logic
    hands = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    assert trade_logic.would_match_human(current_player=0, action=260,
                                          hands=hands, human_seat=1) is True
    assert trade_logic.would_match_human(current_player=0, action=260,
                                          hands=hands, human_seat=2) is False
