from pathlib import Path

from catan_mcts.experiments.e3_rollout_policy import main


def test_e3_smoke_run(tmp_path: Path):
    # Smoke ONLY — see e1 test for sizing rationale.
    out = main(out_root=tmp_path, num_games=1, sims_grid=[2], seed_base=3000)
    import pyarrow.parquet as pq
    games_df = pq.read_table(out / "games.parquet").to_pandas()
    # 2 rollout policies * 1 sim budget * 1 game = 2 game rows
    assert len(games_df) == 2
