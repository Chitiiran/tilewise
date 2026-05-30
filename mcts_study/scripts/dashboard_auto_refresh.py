"""Background daemon: re-aggregate tournament parquets and update dashboard
JSONs every N seconds so partial in-flight progress shows up in the dashboard.

Cited rationale: grid_pass3_tournament.py only writes the dashboard JSON at
cell-level state transitions (cited grid_pass3_tournament.py:_write_cell calls).
While a cell is mid-flight (e.g. h128_l4 partway through 120 games), the
dashboard shows stale data. This daemon polls the parquets on disk and pushes
fresh aggregates to the dashboard JSONs every interval.

Usage:
    python scripts/dashboard_auto_refresh.py [--interval 60]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

# Re-export the aggregation logic from scratch_partial_results.
# The script lives at the project root; we add it to path.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Import the aggregation + dashboard-write functions from the scratch script.
import scratch_partial_results as agg


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--interval", type=float, default=60.0,
                   help="Seconds between refresh cycles (default 60)")
    p.add_argument("--max-iters", type=int, default=0,
                   help="Stop after this many iterations (0 = run forever)")
    args = p.parse_args()

    print(f"[refresh] starting; interval={args.interval}s",
          flush=True)
    iters = 0
    while True:
        try:
            iters += 1
            t0 = time.time()
            agg.update_dashboard("lastepoch", agg.DASH / "grid_pass100k_lastepoch.json")
            agg.update_dashboard("best", agg.DASH / "grid_pass100k_best.json")
            agg.merge_into_main_dashboard()
            elapsed = time.time() - t0
            print(f"[refresh] iter {iters} done in {elapsed:.2f}s",
                  flush=True)
        except Exception as e:
            print(f"[refresh] iter {iters} ERROR: {e}", flush=True)
            traceback.print_exc()
        if args.max_iters and iters >= args.max_iters:
            print(f"[refresh] hit max-iters={args.max_iters}; exiting", flush=True)
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
