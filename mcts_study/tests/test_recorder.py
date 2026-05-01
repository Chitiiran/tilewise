import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from catan_mcts import ACTION_SPACE_SIZE
from catan_mcts.recorder import SelfPlayRecorder, SCHEMA_VERSION


def test_schema_version_is_constant():
    assert SCHEMA_VERSION == 2  # bump explicitly when the schema changes


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
        game_rec.finalize(winner=0, final_vp=[10, 5, 4, 3], length_in_moves=1, action_history=[])
    rec.flush()

    moves = pq.read_table(tmp_path / "moves.parquet").to_pandas()
    games = pq.read_table(tmp_path / "games.parquet").to_pandas()

    assert set(moves.columns) == {
        "seed", "move_index", "current_player", "legal_action_mask",
        "mcts_visit_counts", "action_taken", "mcts_root_value", "schema_version",
    }
    assert set(games.columns) == {
        "seed", "winner", "final_vp", "length_in_moves", "mcts_config_id",
        "action_history", "timed_out", "schema_version",
    }
    assert moves["schema_version"].iloc[0] == SCHEMA_VERSION
    assert games["schema_version"].iloc[0] == SCHEMA_VERSION
    assert len(moves["legal_action_mask"].iloc[0]) == ACTION_SPACE_SIZE
    assert games["winner"].iloc[0] == 0


def test_recorder_writes_config_json(tmp_path: Path):
    rec = SelfPlayRecorder(out_dir=tmp_path, config={"experiment": "smoke", "uct_c": 1.4})
    with rec.game(seed=1) as g:
        g.finalize(winner=-1, final_vp=[0, 0, 0, 0], length_in_moves=0, action_history=[])
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


def test_recorder_checkpoint_writes_labeled_shard(tmp_path: Path):
    """v2 hardening: checkpoint(label) flushes accumulated rows to a labeled
    shard (`moves.<label>.parquet` + `games.<label>.parquet`) and clears the
    in-memory buffers. Lets us flush per cell instead of only at end of main()."""
    rec = SelfPlayRecorder(out_dir=tmp_path, config={"experiment": "smoke"})

    # First cell: one game.
    with rec.game(seed=1) as g:
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool); mask[204] = True
        g.record_move(
            current_player=0, move_index=0,
            legal_action_mask=mask,
            mcts_visit_counts=np.zeros(ACTION_SPACE_SIZE, dtype=np.int32),
            action_taken=204, mcts_root_value=0.0,
        )
        g.finalize(winner=0, final_vp=[10,5,4,3], length_in_moves=1, action_history=[])
    rec.checkpoint("sims=5")

    # After checkpoint, shard files exist and buffers are clear.
    assert (tmp_path / "moves.sims=5.parquet").exists()
    assert (tmp_path / "games.sims=5.parquet").exists()
    games = pq.read_table(tmp_path / "games.sims=5.parquet").to_pandas()
    moves = pq.read_table(tmp_path / "moves.sims=5.parquet").to_pandas()
    assert len(games) == 1 and len(moves) == 1

    # Second cell: another game. Must NOT include data from the first cell.
    with rec.game(seed=2) as g:
        g.finalize(winner=1, final_vp=[5,10,4,3], length_in_moves=1, action_history=[])
    rec.checkpoint("sims=25")

    games2 = pq.read_table(tmp_path / "games.sims=25.parquet").to_pandas()
    moves2 = pq.read_table(tmp_path / "moves.sims=25.parquet").to_pandas()
    assert len(games2) == 1
    assert int(games2["seed"].iloc[0]) == 2
    assert len(moves2) == 0  # no record_move calls in the second game

    # Final flush() with empty buffers should not crash and not overwrite shards.
    rec.flush()
    # Shard files still exist and unchanged.
    assert (tmp_path / "games.sims=5.parquet").exists()


def test_recorder_skip_game_writes_csv_sidecar(tmp_path: Path):
    """v2: skip_game(seed, reason) appends to skipped.csv. Lets experiments
    record timed-out games without polluting moves/games parquet."""
    rec = SelfPlayRecorder(out_dir=tmp_path, config={})
    rec.skip_game(seed=1, reason="wall-clock-timeout", length_in_moves=15234)
    rec.skip_game(seed=7, reason="rollout-cap-pathology", length_in_moves=8401)

    csv_path = tmp_path / "skipped.csv"
    assert csv_path.exists()
    text = csv_path.read_text()
    # Header + 2 rows
    lines = [l for l in text.strip().split("\n") if l]
    assert len(lines) == 3
    assert lines[0].startswith("seed,reason,length_in_moves")
    assert "1,wall-clock-timeout,15234" in lines[1]
    assert "7,rollout-cap-pathology,8401" in lines[2]


def test_finalize_writes_per_seed_shard_immediately(tmp_path: Path):
    """v2 salvage: finalize() must write a per-seed parquet shard on disk
    BEFORE checkpoint/flush. Killing the process after finalize must not
    lose data."""
    rec = SelfPlayRecorder(out_dir=tmp_path, config={})
    with rec.game(seed=42) as g:
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool); mask[204] = True
        g.record_move(
            current_player=0, move_index=0,
            legal_action_mask=mask,
            mcts_visit_counts=np.zeros(ACTION_SPACE_SIZE, dtype=np.int32),
            action_taken=204, mcts_root_value=0.0,
        )
        g.finalize(winner=0, final_vp=[10,5,4,3], length_in_moves=1, action_history=[100,101])

    # Per-seed shards exist immediately, BEFORE any explicit flush/checkpoint.
    games_shard = tmp_path / "games.seed=42.parquet"
    moves_shard = tmp_path / "moves.seed=42.parquet"
    assert games_shard.exists(), "finalize() must write per-seed games shard"
    assert moves_shard.exists(), "finalize() must write per-seed moves shard"
    games = pq.read_table(games_shard).to_pandas()
    assert games["seed"].iloc[0] == 42
    assert list(games["action_history"].iloc[0]) == [100, 101]


def test_skip_game_keeps_action_history_and_moves(tmp_path: Path):
    """v2 salvage: skip_game must persist whatever action_history + moves
    rows the engine already produced, so timed-out games (often the most
    interesting ones) are still inspectable."""
    rec = SelfPlayRecorder(out_dir=tmp_path, config={})
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool); mask[204] = True
    with rec.game(seed=99) as g:
        g.record_move(
            current_player=0, move_index=0,
            legal_action_mask=mask,
            mcts_visit_counts=np.zeros(ACTION_SPACE_SIZE, dtype=np.int32),
            action_taken=204, mcts_root_value=0.0,
        )
        # Caller hands the partial action_history + moves to skip_game.
        rec.skip_game(
            seed=99, reason="wall-clock-timeout", length_in_moves=12345,
            action_history=[1, 2, 3, 4, 5],
            moves_recorder=g,
        )

    # CSV row still appended for the skip log.
    csv_path = tmp_path / "skipped.csv"
    assert csv_path.exists()
    assert "99,wall-clock-timeout,12345" in csv_path.read_text()

    # Per-seed parquet shard written, with timed_out=True flag.
    games_shard = tmp_path / "games.seed=99.parquet"
    assert games_shard.exists()
    games = pq.read_table(games_shard).to_pandas()
    assert games["seed"].iloc[0] == 99
    assert bool(games["timed_out"].iloc[0]) is True
    assert list(games["action_history"].iloc[0]) == [1, 2, 3, 4, 5]
    assert games["winner"].iloc[0] == -1
    moves_shard = tmp_path / "moves.seed=99.parquet"
    assert moves_shard.exists()
    moves = pq.read_table(moves_shard).to_pandas()
    assert len(moves) == 1


def test_checkpoint_compacts_per_seed_shards(tmp_path: Path):
    """v2: checkpoint(label) compacts per-seed shards into one labeled shard
    and removes the per-seed files. Backwards-compatible API for callers."""
    rec = SelfPlayRecorder(out_dir=tmp_path, config={})
    for seed in [1, 2, 3]:
        with rec.game(seed=seed) as g:
            g.finalize(winner=0, final_vp=[10,5,4,3], length_in_moves=1,
                       action_history=[seed * 10])

    # Three per-seed shards on disk before checkpoint.
    assert (tmp_path / "games.seed=1.parquet").exists()
    assert (tmp_path / "games.seed=2.parquet").exists()
    assert (tmp_path / "games.seed=3.parquet").exists()

    rec.checkpoint("sims=5")

    # Combined shard exists and contains all 3 games.
    combined = pq.read_table(tmp_path / "games.sims=5.parquet").to_pandas()
    assert sorted(combined["seed"].tolist()) == [1, 2, 3]
    # Per-seed shards removed (compacted away).
    assert not (tmp_path / "games.seed=1.parquet").exists()
    assert not (tmp_path / "games.seed=2.parquet").exists()
    assert not (tmp_path / "games.seed=3.parquet").exists()


def test_recorder_done_seeds_round_trip(tmp_path: Path):
    """v2: done_seeds() reads done.txt, mark_done(seed) appends. Used by
    experiments to skip already-finished seeds on restart."""
    rec = SelfPlayRecorder(out_dir=tmp_path, config={})
    assert rec.done_seeds() == set()

    rec.mark_done(42)
    rec.mark_done(99)
    assert rec.done_seeds() == {42, 99}

    # New recorder instance over same dir reads the same done.txt
    rec2 = SelfPlayRecorder(out_dir=tmp_path, config={"foo": "bar"})
    assert rec2.done_seeds() == {42, 99}
    rec2.mark_done(7)
    assert rec2.done_seeds() == {42, 99, 7}


def test_recorder_checkpoint_empty_is_noop(tmp_path: Path):
    """checkpoint() with no buffered rows must not write empty shard files."""
    rec = SelfPlayRecorder(out_dir=tmp_path, config={})
    rec.checkpoint("sims=5")
    assert not (tmp_path / "moves.sims=5.parquet").exists()
    assert not (tmp_path / "games.sims=5.parquet").exists()


def test_extract_visit_counts_from_mcts_root():
    """OpenSpiel's MCTSBot exposes the search tree root via `bot.mcts_search(state)`.
    The recorder needs a helper that converts that to a fixed-width visit-count array."""
    import numpy as np
    from open_spiel.python.algorithms import mcts as os_mcts
    from catan_mcts.adapter import CatanGame
    from catan_mcts.recorder import visit_counts_from_root

    game = CatanGame()
    state = game.new_initial_state(seed=42)
    while state.is_chance_node():
        state.apply_action(int(state.chance_outcomes()[0][0]))

    rng = np.random.default_rng(seed=42)
    evaluator = os_mcts.RandomRolloutEvaluator(n_rollouts=1, random_state=rng)
    bot = os_mcts.MCTSBot(
        game=game, uct_c=1.4, max_simulations=5, evaluator=evaluator,
        solve=False, random_state=rng,
    )
    root = bot.mcts_search(state)
    visits = visit_counts_from_root(root)
    assert visits.shape == (ACTION_SPACE_SIZE,)
    assert visits.sum() > 0
    # All non-zero entries correspond to legal actions
    legal = set(state.legal_actions())
    for a, v in enumerate(visits):
        if v > 0:
            assert a in legal
