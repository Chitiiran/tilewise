"""Sliding-window replay buffer (spec §2 BUFFER).

Canonical AZ trains on a window of RECENT self-play games, not all history —
old games reflect an older (weaker) policy and drag the target distribution
backward. The window is selected at run-dir granularity (train_main consumes
whole run dirs); newest-first dirs are taken until the window is filled.

No torch imports — pure pandas bookkeeping, cheap to unit test.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def count_games(run_dir: Path) -> int:
    """Total finished games across a run dir's games*.parquet shards."""
    total = 0
    for shard in Path(run_dir).glob("games*.parquet"):
        total += len(pd.read_parquet(shard, columns=["seed"]))
    return total


def select_window(iter_dirs_newest_first: list[Path], window_games: int) -> list[Path]:
    """Take newest-first run dirs until their game counts sum >= window_games.

    Caller supplies dirs ordered newest-first (the loop knows iteration
    order). Empty dirs are skipped. Raises if no dir contains any game —
    training on nothing should fail loudly at selection, not mid-train.
    """
    selected: list[Path] = []
    total = 0
    for d in iter_dirs_newest_first:
        n = count_games(d)
        if n == 0:
            continue
        selected.append(d)
        total += n
        if total >= window_games:
            break
    if not selected:
        raise ValueError(f"buffer: no games in any of {len(iter_dirs_newest_first)} run dirs")
    return selected
