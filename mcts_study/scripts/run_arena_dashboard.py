"""Launch the AZ arena dashboard (catan_az.dashboard.server) on a port.

Replaces the lost /tmp/prod_dash.py (a tmp copy cleared on WSL restart).
Kept in scripts/ so it survives restarts.

Usage:
    python scripts/run_arena_dashboard.py [--loop-root ...] [--port 8099]
"""
from __future__ import annotations

import argparse

import uvicorn

from catan_az.dashboard.server import create_dashboard


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--loop-root",
                   default="/home/chitii/catan_data/runs/v3/az_loop")
    p.add_argument("--port", type=int, default=8099)
    a = p.parse_args()
    app = create_dashboard(loop_root=a.loop_root, web_port=a.port)
    uvicorn.run(app, host="0.0.0.0", port=a.port, log_level="info")


if __name__ == "__main__":
    main()
