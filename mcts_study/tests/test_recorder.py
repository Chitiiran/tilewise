import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from catan_mcts import ACTION_SPACE_SIZE
from catan_mcts.recorder import SelfPlayRecorder, SCHEMA_VERSION


def test_schema_version_is_constant():
    assert SCHEMA_VERSION == 1  # bump explicitly when the schema changes


def test_recorder_writes_expected_columns(tmp_path: Path):
    rec = SelfPlayRecorder(out_dir=tmp_path, config={"experiment": "smoke"})
    with rec.game(seed=42) as game_rec:
        # Fake one move (with a legal mask that allows action 204)
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
        mask[204] = True  # end turn — legal in Main phase
        game_rec.record_move(
            current_player=0,
            move_index=0,
            legal_action_mask=mask,
            mcts_visit_counts=np.zeros(ACTION_SPACE_SIZE, dtype=np.int32),
            action_taken=204,
            mcts_root_value=0.5,
        )
        game_rec.finalize(winner=0, final_vp=[10, 5, 4, 3], length_in_moves=1)
    rec.flush()

    moves = pq.read_table(tmp_path / "moves.parquet").to_pandas()
    games = pq.read_table(tmp_path / "games.parquet").to_pandas()

    assert set(moves.columns) == {
        "seed", "move_index", "current_player", "legal_action_mask",
        "mcts_visit_counts", "action_taken", "mcts_root_value", "schema_version",
    }
    assert set(games.columns) == {
        "seed", "winner", "final_vp", "length_in_moves", "mcts_config_id", "schema_version",
    }
    assert moves["schema_version"].iloc[0] == SCHEMA_VERSION
    assert games["schema_version"].iloc[0] == SCHEMA_VERSION
    assert len(moves["legal_action_mask"].iloc[0]) == ACTION_SPACE_SIZE
    assert games["winner"].iloc[0] == 0


def test_recorder_writes_config_json(tmp_path: Path):
    rec = SelfPlayRecorder(out_dir=tmp_path, config={"experiment": "smoke", "uct_c": 1.4})
    with rec.game(seed=1) as g:
        g.finalize(winner=-1, final_vp=[0, 0, 0, 0], length_in_moves=0)
    rec.flush()

    cfg = json.loads((tmp_path / "config.json").read_text())
    assert cfg["experiment"] == "smoke"
    assert cfg["uct_c"] == 1.4
    assert cfg["schema_version"] == SCHEMA_VERSION


def test_record_move_rejects_action_not_in_legal_mask(tmp_path: Path):
    rec = SelfPlayRecorder(out_dir=tmp_path, config={})
    with rec.game(seed=42) as game_rec:
        try:
            game_rec.record_move(
                current_player=0,
                move_index=0,
                legal_action_mask=np.zeros(ACTION_SPACE_SIZE, dtype=bool),  # no legal actions
                mcts_visit_counts=np.zeros(ACTION_SPACE_SIZE, dtype=np.int32),
                action_taken=42,
                mcts_root_value=0.0,
            )
        except AssertionError:
            return
    raise AssertionError("expected record_move to reject action with mask=False")
