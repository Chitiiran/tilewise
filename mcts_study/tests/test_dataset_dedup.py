"""Loader-side dedup of duplicated move shards (run-dir collision guard).

The 2026-06-11 iteration-1 self-play hit a make_run_dir minute-collision:
several procs wrote into one dir and their racing end-of-run consolidations
duplicated rows exactly. games-table dup collapses harmlessly (dicts keyed
by seed), but moves-table dup would double-weight those positions in
training. CatanReplayDataset must dedup moves by (seed, move_index,
current_player) so a collided corpus trains correctly without mutating the
on-disk data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _write_game(d, seed, n_moves):
    d.mkdir(parents=True, exist_ok=True)
    hist = list(range(n_moves * 2))   # arbitrary non-chance action ids
    pd.DataFrame({
        "seed": [seed], "winner": [0],
        "final_vp": [[10, 5, 4, 3]], "length_in_moves": [n_moves],
        "mcts_config_id": ["abc"], "action_history": [hist],
        "timed_out": [False], "schema_version": [2],
    }).to_parquet(d / f"games.seed={seed}.shard.parquet")
    rows = []
    for mi in range(n_moves):
        rows.append({
            "seed": seed, "move_index": mi, "current_player": 0,
            "legal_action_mask": np.ones(280, dtype=bool),
            "mcts_visit_counts": np.ones(280, dtype=np.int64),
            "action_taken": 0, "mcts_root_value": 0.0, "schema_version": 2,
        })
    return pd.DataFrame(rows)


def test_duplicate_move_shards_deduped(tmp_path):
    from catan_gnn.dataset import CatanReplayDataset
    d = tmp_path / "collided"
    moves = _write_game(d, seed=100, n_moves=5)
    # Simulate the collision: SAME moves written to two differently-named shards.
    moves.to_parquet(d / "moves.a.parquet")
    moves.to_parquet(d / "moves.b.parquet")

    ds = CatanReplayDataset([d])
    # Without dedup this would be 10; with dedup it must be 5 (the true count).
    assert len(ds) == 5


def test_distinct_moves_not_collapsed(tmp_path):
    from catan_gnn.dataset import CatanReplayDataset
    d = tmp_path / "clean"
    m1 = _write_game(d, seed=200, n_moves=3)
    m2 = _write_game(d, seed=201, n_moves=4)
    m1.to_parquet(d / "moves.a.parquet")
    m2.to_parquet(d / "moves.b.parquet")
    ds = CatanReplayDataset([d])
    assert len(ds) == 7   # 3 + 4, nothing dropped
