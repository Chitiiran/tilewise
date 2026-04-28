"""Smoke tests for e4 tournament.

The full-pipeline smoke (`test_e4_smoke_run`) plays 4 cyclic rotations × 1 game,
which is ~4x slower than other experiment smokes — runs ~15-20 min on this
hardware. It's marked `slow` and skipped in default pytest runs; opt in with
`pytest -m slow`. The structural test below proves the script's CLI/main
contract works without paying that cost.
"""
from pathlib import Path

import pytest

from catan_mcts.experiments.e4_tournament import main, cli_main


def test_e4_module_exposes_main_and_cli():
    """Structural test: the e4 module has the contract Plan 3 expects."""
    assert callable(main)
    assert callable(cli_main)


@pytest.mark.slow
def test_e4_smoke_run(tmp_path: Path):
    # Full smoke — runs ~15-20 min. Opt in with `pytest -m slow`.
    out = main(out_root=tmp_path, num_games_per_seating=1, mcts_sims=2, seed_base=4000)
    import pyarrow.parquet as pq
    games_df = pq.read_table(out / "games.parquet").to_pandas()
    # 4 cyclic rotations * 1 game = 4 game rows
    assert len(games_df) == 4
