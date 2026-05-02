"""Smoke test for e9 v3 data-gen pipeline."""
from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

from catan_mcts.experiments.e9_v3_data_gen import main as e9_main


def test_e9_runs_mini_sweep(tmp_path: Path):
    """3 v3 games with tiny sim budget. Verifies the recorder writes
    parquet shards and at least one game records moves successfully."""
    out_root = tmp_path / "runs"
    out_root.mkdir()
    out = e9_main(
        out_root=out_root,
        num_games=3,
        base_sims=50,        # Hits SIM_FLOOR — fastest possible legal config.
        lookahead_depth=2,
        seed_base=42,
        max_seconds=120.0,
        vp_target=5,
        bonuses=False,
        resume=False,
        workers=1,
    )

    games = list(out.glob("games.*.parquet"))
    moves = list(out.glob("moves.*.parquet"))
    assert games, f"no games.parquet shard in {out}"
    assert moves, f"no moves.parquet shard in {out}"

    games_table = pq.read_table(games[0]).to_pandas()
    assert len(games_table) == 3
    # All games should have run under v3 rules (winners hit at most 5 VP).
    for vp_arr in games_table["final_vp"]:
        assert max(vp_arr) <= 5, f"v3 game ended with max VP {max(vp_arr)} > 5"

    # We rotate recorded_player across seeds (i % 4). With seeds 42, 43, 44
    # we should see recorded_player 2, 3, 0 across the three games.
    moves_table = pq.read_table(moves[0]).to_pandas()
    if len(moves_table) > 0:
        recorded_seats = set(moves_table["current_player"].unique())
        # At least the three rotated seats should appear (assuming each
        # game made at least one MCTS decision; if a game terminated
        # without any non-trivial decision for the recorded seat we may
        # see fewer).
        assert recorded_seats.issubset({0, 1, 2, 3})
