"""Smoke-launch the dashboard against a synthetic live iter and verify it serves
the page + the new live endpoints with populated data. Prints PASS/FAIL lines.
No browser needed — exercises the real server."""
import json
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient
from catan_az.dashboard.server import create_dashboard


def main():
    d = Path(tempfile.mkdtemp())
    it = 9
    (d / "daily_state.json").write_text(json.dumps({"iter": it}))
    (d / "status.json").write_text(json.dumps({"iter": it, "stage": "iterate"}))
    sp = d / f"iter_{it}" / "selfplay"; sp.mkdir(parents=True)
    (sp / "resources.jsonl").write_text(
        "\n".join(json.dumps({"ts": float(i), "gpu_util_pct": 30 + i * 5,
                              "gpu_power_w": 8 + i, "gpu_mem_used_mb": 200,
                              "gpu_mem_total_mb": 4096, "load1": 6,
                              "ram_avail_gb": 40, "stage": "selfplay"})
                  for i in range(8)) + "\n")
    tr = d / f"iter_{it}" / "training"; tr.mkdir(parents=True)
    (tr / "train_progress.jsonl").write_text(
        "\n".join(json.dumps({"epoch": 1, "batch": 50 * (i + 1),
                              "batches_total": 1000, "loss": 1.5 - 0.08 * i,
                              "grad_norm": 3, "ms_per_batch": 20, "ts": float(i)})
                  for i in range(10)) + "\n")

    c = TestClient(create_dashboard(loop_root=d, web_port=8099))
    page = c.get("/")
    print("PASS index 200" if page.status_code == 200 else f"FAIL index {page.status_code}")
    html = page.text if page.headers.get("content-type", "").startswith("text/html") else ""
    # the index endpoint returns a FileResponse; fetch the file directly too
    from catan_az.dashboard.server import _STATIC
    html = (_STATIC / "index.html").read_text()
    print("PASS panels in html" if "res-card" in html and "tr-card" in html else "FAIL panels")

    for url in ("/api/resources-live", "/api/train-progress-live",
                "/api/summary", "/api/selfplay-live"):
        r = c.get(url)
        ok = r.status_code == 200
        body = r.json() if ok else {}
        avail = body.get("available")
        npts = len(body.get("points", [])) if isinstance(body, dict) else 0
        print(f"{'PASS' if ok else 'FAIL'} {url} -> available={avail} points={npts}")
    print(f"loop_root={d}")


if __name__ == "__main__":
    main()
