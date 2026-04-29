"""Tests for SelfPlayRecorder v2 schema (adds action_history)."""
from pathlib import Path

import pyarrow.parquet as pq

from catan_mcts.recorder import SCHEMA_VERSION, SelfPlayRecorder


def test_schema_version_is_2():
    assert SCHEMA_VERSION == 2


def test_games_parquet_has_action_history(tmp_path: Path):
    rec = SelfPlayRecorder(tmp_path, config={"experiment": "test"})
    with rec.game(seed=42) as g:
        g.finalize(
            winner=0,
            final_vp=[10, 5, 4, 3],
            length_in_moves=100,
            action_history=[1, 2, 0x80000004, 5],   # mix of decisions + chance bits
        )
    rec.flush()

    games = pq.read_table(tmp_path / "games.parquet").to_pandas()
    assert len(games) == 1
    assert list(games["action_history"].iloc[0]) == [1, 2, 0x80000004, 5]
    assert int(games["schema_version"].iloc[0]) == 2
