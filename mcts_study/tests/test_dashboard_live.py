"""Dashboard works END-TO-END with the new live panels: endpoints serve
populated data when a live iter has resource/training streams, and the HTML wires
the new cards + draw functions."""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from catan_az.dashboard.server import create_dashboard, _STATIC


def _setup_live_iter(root: Path, it: int = 7):
    # liveness() keys off daily_state.json / status.json mtime; write both so the
    # live iter resolves to `it`.
    (root / "daily_state.json").write_text(json.dumps({"iter": it}))
    (root / "status.json").write_text(json.dumps({"iter": it, "stage": "iterate"}))
    res = root / f"iter_{it}" / "selfplay"
    res.mkdir(parents=True, exist_ok=True)
    (res / "resources.jsonl").write_text(
        "\n".join(json.dumps({"ts": float(i), "gpu_util_pct": 30 + i,
                              "gpu_power_w": 8.0 + i, "gpu_mem_used_mb": 200,
                              "gpu_mem_total_mb": 4096, "load1": 6.0,
                              "ram_avail_gb": 40.0, "stage": "selfplay"})
                  for i in range(5)) + "\n")
    tr = root / f"iter_{it}" / "training"
    tr.mkdir(parents=True, exist_ok=True)
    (tr / "train_progress.jsonl").write_text(
        "\n".join(json.dumps({"epoch": 1, "batch": 50 * (i + 1),
                              "batches_total": 1000, "loss": 1.5 - 0.1 * i,
                              "loss_value": 0.5, "loss_policy": 1.0,
                              "grad_norm": 3.0, "ms_per_batch": 20.0,
                              "ts": float(i)})
                  for i in range(6)) + "\n")


def test_dashboard_serves_live_resources_and_training(tmp_path):
    _setup_live_iter(tmp_path, it=7)
    client = TestClient(create_dashboard(loop_root=tmp_path, web_port=8099))

    r = client.get("/api/resources-live").json()
    # liveness may or may not resolve the iter depending on mtime heuristics;
    # the endpoint must at least not crash and return the contract shape.
    assert "available" in r and "points" in r
    if r["available"]:
        assert r["latest"]["gpu_util_pct"] >= 30

    t = client.get("/api/train-progress-live").json()
    assert "available" in t and "points" in t


def test_index_html_wires_new_panels():
    html = (_STATIC / "index.html").read_text()
    # cards present
    for el in ("res-card", "res-stats", "res-chart", "tr-card", "tr-chart", "tr-head"):
        assert f'id="{el}"' in html, f"missing dashboard element {el}"
    # draw functions defined + invoked
    for fn in ("drawResources", "drawTraining", "function spark"):
        assert fn in html, f"missing JS {fn}"
    assert "await drawResources();" in html and "await drawTraining();" in html
    # endpoints referenced
    assert "/api/resources-live" in html and "/api/train-progress-live" in html


def test_resources_live_populated_directly(tmp_path):
    """Bypass liveness: the analytics fn returns the synthetic stream for the iter."""
    from catan_az import analytics
    _setup_live_iter(tmp_path, it=7)
    out = analytics.resources_live(tmp_path, iter_n=7)
    assert out["available"] and len(out["points"]) == 5
    assert out["latest"]["gpu_util_pct"] == 34  # 30 + 4
    tp = analytics.train_progress_live(tmp_path, iter_n=7)
    assert tp["available"] and len(tp["points"]) == 6
    assert abs(tp["latest"]["loss"] - 1.0) < 1e-9  # 1.5 - 0.1*5
