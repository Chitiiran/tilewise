from pathlib import Path

from catan_mcts.experiments.e2_ucb_c_sweep import main


def test_e2_smoke_run(tmp_path: Path):
    # Smoke ONLY — see e1 test for rationale on tiny params.
    out = main(
        out_root=tmp_path, num_games=1, c_grid=[0.5, 4.0], sims=2, seed_base=2000,
    )
    import pyarrow.parquet as pq
    games_df = pq.read_table(out / "games.parquet").to_pandas()
    # 2 c-values * 1 game = 2
    assert len(games_df) == 2
