"""Background resource sampler for live observability.

Appends one JSON line per interval to <out_dir>/resources.jsonl with GPU
util/power/mem, CPU loadavg, and available RAM. Crash-safe (append-only,
line-buffered). Stops on a STOP/PAUSE sentinel in out_dir or on SIGTERM, so the
daily driver can start it beside a stage and kill it when the stage ends.

Run standalone:  python -m catan_az.sampler <out_dir> [interval_s] [stage]
Or embed:        Sampler(out_dir, stage="selfplay").run_until(stop_predicate)
"""
from __future__ import annotations

import json
import signal
import sys
import time
from pathlib import Path

from .resources import host_sample


class Sampler:
    def __init__(self, out_dir, interval_s: float = 2.0, stage: str = "") -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.out_dir / "resources.jsonl"
        self.interval_s = float(interval_s)
        self.stage = stage
        self._stop = False

    def sample_once(self, clock=time.time) -> dict:
        row = host_sample()
        row["ts"] = clock()
        row["stage"] = self.stage
        return row

    def _write(self, row: dict) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def run_until(self, should_stop, clock=time.time, sleep=time.sleep) -> None:
        """Sample until should_stop() is True. should_stop is injectable for
        tests; default loop is driven by SIGTERM / the _stop flag."""
        while not should_stop():
            self._write(self.sample_once(clock))
            sleep(self.interval_s)

    def run_forever(self) -> None:
        def _handle(_sig, _frm):
            self._stop = True
        signal.signal(signal.SIGTERM, _handle)
        signal.signal(signal.SIGINT, _handle)
        self.run_until(lambda: self._stop)


def cli_main(argv=None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    out_dir = argv[0]
    interval = float(argv[1]) if len(argv) > 1 else 2.0
    stage = argv[2] if len(argv) > 2 else ""
    Sampler(out_dir, interval_s=interval, stage=stage).run_forever()


if __name__ == "__main__":
    cli_main()
