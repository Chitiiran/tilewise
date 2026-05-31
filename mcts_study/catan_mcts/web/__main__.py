"""Launch the play-vs-bots server.

Usage (WSL, mcts-study venv active, from mcts_study/):
    python -m catan_mcts.web --checkpoints-dir /path/to/checkpoints \
                             --replays-dir /path/to/replays --port 8000
Then open http://localhost:8000 in the Windows browser.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from catan_mcts.web.server import create_app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints-dir", type=Path, default=Path("."),
                    help="dir scanned recursively for *.pt GNN checkpoints")
    ap.add_argument("--replays-dir", type=Path, default=Path("."),
                    help="dir scanned for existing playback_seed_*/index.html replays")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    app = create_app(checkpoints_dir=args.checkpoints_dir, replays_dir=args.replays_dir)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
