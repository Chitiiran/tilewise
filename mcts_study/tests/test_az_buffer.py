"""Sliding-window replay buffer over parquet run dirs (spec §2 BUFFER)."""
from __future__ import annotations

import pandas as pd
import pytest


def _mk_run_dir(root, name, n_games):
    d = root / name
    d.mkdir(parents=True)
    if n_games:
        df = pd.DataFrame({
            "seed": range(n_games),
            "winner": [i % 4 for i in range(n_games)],
        })
        df.to_parquet(d / "games.abc123.parquet")
    return d


def test_count_games_sums_shards(tmp_path):
    from catan_az.buffer import count_games
    d = tmp_path / "run"
    d.mkdir()
    pd.DataFrame({"seed": [1, 2], "winner": [0, 1]}).to_parquet(d / "games.a.parquet")
    pd.DataFrame({"seed": [3], "winner": [2]}).to_parquet(d / "games.b.parquet")
    assert count_games(d) == 3


def test_count_games_empty_dir(tmp_path):
    from catan_az.buffer import count_games
    d = tmp_path / "empty"
    d.mkdir()
    assert count_games(d) == 0


def test_window_smaller_than_one_dir(tmp_path):
    from catan_az.buffer import select_window
    d1 = _mk_run_dir(tmp_path, "old", 100)
    d2 = _mk_run_dir(tmp_path, "new", 100)
    # newest-first list order is the caller's contract
    sel = select_window([d2, d1], window_games=50)
    assert sel == [d2]


def test_window_spans_dirs_newest_first(tmp_path):
    from catan_az.buffer import select_window
    d1 = _mk_run_dir(tmp_path, "oldest", 100)
    d2 = _mk_run_dir(tmp_path, "mid", 100)
    d3 = _mk_run_dir(tmp_path, "newest", 100)
    sel = select_window([d3, d2, d1], window_games=150)
    assert sel == [d3, d2]


def test_window_exceeding_total_takes_all(tmp_path):
    from catan_az.buffer import select_window
    d1 = _mk_run_dir(tmp_path, "a", 10)
    d2 = _mk_run_dir(tmp_path, "b", 10)
    sel = select_window([d2, d1], window_games=10_000)
    assert sel == [d2, d1]


def test_empty_dirs_skipped(tmp_path):
    from catan_az.buffer import select_window
    d1 = _mk_run_dir(tmp_path, "full", 100)
    d_empty = _mk_run_dir(tmp_path, "empty", 0)
    sel = select_window([d_empty, d1], window_games=50)
    assert sel == [d1]


def test_no_games_anywhere_raises(tmp_path):
    from catan_az.buffer import select_window
    d_empty = _mk_run_dir(tmp_path, "empty", 0)
    with pytest.raises(ValueError, match="no games"):
        select_window([d_empty], window_games=100)
