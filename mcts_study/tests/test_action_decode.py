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


def test_decode_city_target_offset():
    from catan_mcts.web import action_decode
    # City ids are 54..107; target maps back to the vertex (a - 54).
    assert action_decode.decode(54)["kind"] == "build_city"
    assert action_decode.decode(54)["target"] == 0
    assert action_decode.decode(107)["target"] == 53


def test_decode_boundaries():
    from catan_mcts.web import action_decode
    # The off-by-one-prone seams between ranges.
    assert action_decode.decode(198)["kind"] == "move_robber"   # last robber
    assert action_decode.decode(199)["kind"] == "discard"        # first discard
    assert action_decode.decode(259)["kind"] == "play_dev"       # last dev play
    assert action_decode.decode(260)["kind"] == "propose_trade"  # first trade


def test_decode_out_of_range_is_unknown():
    from catan_mcts.web import action_decode
    for a in (280, -1, 999):
        d = action_decode.decode(a)
        assert d["kind"] == "unknown"
        assert d["target"] is None


def test_decode_many():
    from catan_mcts.web import action_decode
    out = action_decode.decode_many([0, 108, 204, 205])
    assert [o["id"] for o in out] == [0, 108, 204, 205]
