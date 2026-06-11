"""Tests for difficulty presets in the web bot registry.

The ladder maps a single difficulty id to a full bot spec so the lobby can
offer "Easy/Medium/Hard" without exposing checkpoints. Tier choices are
justified by measured winrates (see plan 2026-06-10-az-difficulty-bots.md):
no mid-sims GnnMcts tier exists because sims 8-32 measure WORSE than argmax
(the 2026-06-01 cheap-search valley).
"""
from __future__ import annotations

import pytest


def _mk_cell6_checkpoint(root):
    """Create the Cell-6 checkpoint file at its expected relative path."""
    p = root / "training" / "loss_aug" / "06_cand11_cand8_cand10_h128_l4" / \
        "training_h128_l4" / "checkpoint_epoch10.pt"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"fake")
    return p


def test_list_difficulties_returns_ladder():
    from catan_mcts.web import bot_registry
    diffs = bot_registry.list_difficulties()
    ids = [d["id"] for d in diffs]
    assert ids == ["beginner", "easy", "medium", "hard", "expert"]
    for d in diffs:
        assert d["label"]
        assert d["spec"]["type"]


def test_resolve_seat_spec_passthrough_for_explicit_type():
    """Back-compat: a spec with an explicit type is returned unchanged."""
    from catan_mcts.web import bot_registry
    spec = {"type": "Random"}
    assert bot_registry.resolve_seat_spec(spec, checkpoints_dir=None) == spec


def test_resolve_seat_spec_beginner_needs_no_checkpoint():
    from catan_mcts.web import bot_registry
    spec = bot_registry.resolve_seat_spec({"difficulty": "beginner"},
                                          checkpoints_dir=None)
    assert spec["type"] == "Random"


def test_resolve_seat_spec_expert_resolves_checkpoint(tmp_path):
    from catan_mcts.web import bot_registry
    ckpt = _mk_cell6_checkpoint(tmp_path)
    spec = bot_registry.resolve_seat_spec({"difficulty": "expert"},
                                          checkpoints_dir=tmp_path)
    assert spec["type"] == "GnnMcts"
    assert spec["sims"] == 200
    assert spec["checkpoint"] == str(ckpt)


def test_resolve_seat_spec_medium_resolves_checkpoint(tmp_path):
    from catan_mcts.web import bot_registry
    ckpt = _mk_cell6_checkpoint(tmp_path)
    spec = bot_registry.resolve_seat_spec({"difficulty": "medium"},
                                          checkpoints_dir=tmp_path)
    assert spec["type"] == "PureGnn"
    assert spec["checkpoint"] == str(ckpt)


def test_resolve_seat_spec_missing_checkpoint_raises(tmp_path):
    """Empty checkpoints root -> clear error naming the expected rel path."""
    from catan_mcts.web import bot_registry
    with pytest.raises(ValueError, match="checkpoint_epoch10.pt"):
        bot_registry.resolve_seat_spec({"difficulty": "expert"},
                                       checkpoints_dir=tmp_path)


def test_resolve_seat_spec_unknown_difficulty_raises():
    from catan_mcts.web import bot_registry
    with pytest.raises(ValueError, match="unknown difficulty"):
        bot_registry.resolve_seat_spec({"difficulty": "impossible"},
                                       checkpoints_dir=None)


def test_resolve_seat_spec_rejects_empty_spec():
    from catan_mcts.web import bot_registry
    with pytest.raises(ValueError, match="type|difficulty"):
        bot_registry.resolve_seat_spec({}, checkpoints_dir=None)


# --- API integration -------------------------------------------------------

@pytest.fixture()
def client(tmp_path):
    from fastapi.testclient import TestClient
    from catan_mcts.web.server import create_app
    app = create_app(checkpoints_dir=tmp_path, replays_dir=tmp_path)
    return TestClient(app)


def test_bots_endpoint_includes_difficulties(client):
    data = client.get("/api/bots").json()
    assert "difficulties" in data
    assert [d["id"] for d in data["difficulties"]] == \
        ["beginner", "easy", "medium", "hard", "expert"]


def test_create_game_with_difficulty_seats(client):
    """Difficulty-only seats must create a playable game (no torch needed
    for beginner/easy/hard)."""
    r = client.post("/api/games", json={
        "human_seat": 0,
        "seats": {
            "1": {"difficulty": "beginner"},
            "2": {"difficulty": "easy"},
            "3": {"difficulty": "hard"},
        },
        "seed": 42,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["game_id"]
    # Real status vocabulary (see game_session/test_web_api): the game must
    # land in a live, non-error state right after creation.
    assert body["state"]["status"] in ("your_turn", "bots_thinking",
                                       "trade_offer", "game_over")


def test_create_game_with_unknown_difficulty_400(client):
    r = client.post("/api/games", json={
        "human_seat": 0,
        "seats": {"1": {"difficulty": "nope"},
                  "2": {"difficulty": "beginner"},
                  "3": {"difficulty": "beginner"}},
        "seed": 1,
    })
    assert r.status_code == 400
