"""Observability sampler: pure parsers + the sampling loop (injected clock/stop),
so it's testable without a GPU."""
import json

from catan_az import resources
from catan_az.sampler import Sampler


def test_parse_gpu_sample():
    out = "34, 7.93, 166, 4096\n"
    s = resources.parse_gpu_sample(out)
    assert s == {"gpu_util_pct": 34.0, "gpu_power_w": 7.93,
                 "gpu_mem_used_mb": 166.0, "gpu_mem_total_mb": 4096.0}


def test_parse_loadavg():
    assert resources.parse_loadavg("0.85 1.20 2.00 1/123 456") == {
        "load1": 0.85, "load5": 1.20, "load15": 2.00}


def test_sample_once_shape(tmp_path):
    s = Sampler(tmp_path, stage="selfplay")
    row = s.sample_once(clock=lambda: 123.0)
    # GPU may be absent in CI -> values may be -1, but keys must exist.
    for k in ("gpu_util_pct", "gpu_power_w", "gpu_mem_used_mb",
              "load1", "ram_avail_gb", "ts", "stage"):
        assert k in row
    assert row["ts"] == 123.0 and row["stage"] == "selfplay"


def test_run_until_writes_lines(tmp_path):
    s = Sampler(tmp_path, interval_s=0.0, stage="train")
    n = {"i": 0}
    def stop():
        n["i"] += 1
        return n["i"] > 3          # write 3 rows then stop
    s.run_until(stop, clock=lambda: 1.0, sleep=lambda _x: None)
    lines = (tmp_path / "resources.jsonl").read_text().splitlines()
    assert len(lines) == 3
    rows = [json.loads(x) for x in lines]
    assert all(r["stage"] == "train" for r in rows)
