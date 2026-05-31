"""Tests for action int -> UI object decoding."""
from __future__ import annotations


def test_decode_settlement():
    from catan_mcts.web import action_decode
    d = action_decode.decode(12)
    assert d["id"] == 12
    assert d["kind"] == "build_settlement"
    assert d["target"] == 12
    assert "Settlement" in d["label"]


def test_decode_road_target_is_edge():
    from catan_mcts.web import action_decode
    d = action_decode.decode(108)  # first road
    assert d["kind"] == "build_road"
    assert d["target"] == 0


def test_decode_move_robber_target_is_hex():
    from catan_mcts.web import action_decode
    d = action_decode.decode(180)
    assert d["kind"] == "move_robber"
    assert d["target"] == 0


def test_decode_non_spatial_has_null_target():
    from catan_mcts.web import action_decode
    for a, kind in [(205, "roll"), (204, "end_turn"), (226, "buy_dev"),
                    (206, "trade_bank"), (260, "propose_trade"), (227, "play_dev")]:
        d = action_decode.decode(a)
        assert d["kind"] == kind, (a, d)
        assert d["target"] is None


def test_decode_many():
    from catan_mcts.web import action_decode
    out = action_decode.decode_many([0, 108, 204, 205])
    assert [o["id"] for o in out] == [0, 108, 204, 205]
