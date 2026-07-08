"""Observability analytics: jsonl tailing + resources/train-progress live views,
and the dashboard endpoints that surface them."""
import json
from pathlib import Path

from catan_az import analytics


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_tail_jsonl_tolerates_torn_line(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"a":1}\n{"a":2}\n{"a": 3,  <- torn\n')
    rows = analytics._tail_jsonl(p)
    assert rows == [{"a": 1}, {"a": 2}]   # torn final line skipped


def test_tail_jsonl_missing(tmp_path):
    assert analytics._tail_jsonl(tmp_path / "nope.jsonl") == []


def test_resources_live_aggregates_and_sorts(tmp_path):
    base = tmp_path / "iter_3"
    _write_jsonl(base / "selfplay" / "resources.jsonl", [
        {"ts": 2.0, "gpu_util_pct": 30, "stage": "selfplay"},
        {"ts": 1.0, "gpu_util_pct": 10, "stage": "selfplay"},
    ])
    out = analytics.resources_live(tmp_path, iter_n=3)
    assert out["available"]
    ts = [r["ts"] for r in out["points"]]
    assert ts == sorted(ts)                      # sorted by ts
    assert out["latest"]["gpu_util_pct"] == 30   # newest


def test_train_progress_live(tmp_path):
    _write_jsonl(tmp_path / "iter_5" / "training" / "train_progress.jsonl", [
        {"batch": 50, "loss": 1.2, "grad_norm": 3.0, "ts": 1.0},
        {"batch": 100, "loss": 0.9, "grad_norm": 2.1, "ts": 2.0},
    ])
    out = analytics.train_progress_live(tmp_path, iter_n=5)
    assert out["available"]
    assert out["latest"]["loss"] == 0.9
    assert len(out["points"]) == 2


def test_dashboard_endpoints_exist(tmp_path):
    """The new endpoints are registered and return the not-running shape when no
    live iter exists (no crash)."""
    from catan_az.dashboard.server import create_dashboard
    from fastapi.testclient import TestClient

    app = create_dashboard(loop_root=tmp_path, web_port=8099)
    client = TestClient(app)
    for url in ("/api/resources-live", "/api/train-progress-live"):
        r = client.get(url)
        assert r.status_code == 200
        assert r.json()["available"] is False
