"""Tests for the extracted engine-state serializer."""
from __future__ import annotations

import json


def test_serialize_state_has_full_field_set():
    from catan_bot import _engine
    from catan_mcts.web import serializers
    eng = _engine.Engine(4242)
    st = serializers.serialize_state(eng, narration="(initial)")
    required = {"n", "cp", "phase", "s", "c", "r", "rh", "vp", "hands",
                "bank", "dev_held", "ports", "lr_len", "knights", "built",
                "lr_holder", "la_holder", "vp_played"}
    assert required.issubset(st.keys())
    json.dumps(st)


def test_action_desc_blocks():
    from catan_mcts.web import serializers
    assert "BuildSettlement" in serializers.action_desc(0)
    assert "ProposeTrade" in serializers.action_desc(260)
    assert serializers.action_desc(204) == "EndTurn"
