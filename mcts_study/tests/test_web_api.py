"""FastAPI endpoint contract tests."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path):
    from catan_mcts.web.server import create_app
    app = create_app(checkpoints_dir=tmp_path, replays_dir=tmp_path)
    return TestClient(app)


def test_bots_endpoint(client):
    r = client.get("/api/bots")
    assert r.status_code == 200
    body = r.json()
    ids = {t["id"] for t in body["types"]}
    assert {"Random", "PureGnn"} <= ids
    assert "checkpoints" in body


def _all_random_setup(human_seat=0):
    return {
        "human_seat": human_seat,
        "seats": {str(s): {"type": "Random"} for s in range(4) if s != human_seat},
        "rules": {"vp_target": 10, "bonuses": True},
        "seed": 4242,
    }


def test_create_and_play_to_terminal(client):
    r = client.post("/api/games", json=_all_random_setup())
    assert r.status_code == 200
    body = r.json()
    gid = body["game_id"]
    assert "board" in body and body["board"]["png_b64"]
    state = body["state"]
    for _ in range(100000):
        if state["status"] == "game_over":
            break
        if state["status"] == "your_turn":
            aid = state["legal_actions"][0]["id"]
            state = client.post(f"/api/games/{gid}/action", json={"action": aid}).json()
        elif state["status"] == "trade_offer":
            state = client.post(f"/api/games/{gid}/trade-response", json={"accept": False}).json()
        else:
            state = client.get(f"/api/games/{gid}/state").json()
    assert state["status"] == "game_over"
    assert state["returns"] is not None


def test_illegal_action_returns_409(client):
    gid = client.post("/api/games", json=_all_random_setup()).json()["game_id"]
    r = client.post(f"/api/games/{gid}/action", json={"action": 9999})
    assert r.status_code == 409


def test_unknown_game_404(client):
    r = client.get("/api/games/does-not-exist/state")
    assert r.status_code == 404


def test_sse_emits_at_least_one_event(client):
    gid = client.post("/api/games", json=_all_random_setup()).json()["game_id"]
    with client.stream("GET", f"/api/games/{gid}/events") as r:
        assert r.status_code == 200
        got = None
        for line in r.iter_lines():
            if line and line.startswith("data:"):
                got = line
                break
        assert got is not None


def test_root_serves_index(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
