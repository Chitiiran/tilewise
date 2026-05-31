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
