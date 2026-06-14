"""Dashboard summary endpoint (spec 2026-06-13 §7)."""
from __future__ import annotations

import json


def test_dashboard_summary_endpoint(tmp_path):
    from fastapi.testclient import TestClient

    from catan_az.dashboard.server import create_dashboard
    (tmp_path / "ladder.json").write_text(json.dumps({
        "champion": "az_iter_1",
        "entries": {"az_iter_1": {"name": "az_iter_1", "checkpoint": "/c.pt",
                                  "elo": 1003.6, "games": 120, "created_iter": 1}},
        "history": []}))
    (tmp_path / "status.json").write_text(json.dumps({"iter": 2, "stage": "arena"}))
    (tmp_path / "journal.csv").write_text(
        "iter,verdict,arena_winrate\n1,promote,0.65\n")
    app = create_dashboard(loop_root=tmp_path, web_port=8000)
    c = TestClient(app)
    r = c.get("/api/summary").json()
    assert r["champion"]["name"] == "az_iter_1"
    assert r["status"]["stage"] == "arena"
    assert len(r["journal"]) == 1
    assert "play_champion_url" in r


def test_dashboard_tolerates_missing_files(tmp_path):
    from fastapi.testclient import TestClient

    from catan_az.dashboard.server import create_dashboard
    app = create_dashboard(loop_root=tmp_path, web_port=8000)
    c = TestClient(app)
    r = c.get("/api/summary").json()
    assert r["champion"] == {} and r["journal"] == []
