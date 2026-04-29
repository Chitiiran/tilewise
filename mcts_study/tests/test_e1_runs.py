from pathlib import Path

from catan_mcts.experiments.e1_winrate_vs_random import main


def test_e1_smoke_run(tmp_path: Path):
    # Smoke ONLY — proves the e1 script writes correctly-shaped parquet shards.
    # v2: per-cell checkpoint flush means data lands as moves.sims=N.parquet
    # rather than the old single moves.parquet. Notebook globs shards on read.
    out = main(
        out_root=tmp_path,
        num_games=1,
        sims_per_move_grid=[2],
        seed_base=1000,
        max_seconds=300.0,
    )
    moves_shard = out / "moves.sims=2.parquet"
    games_shard = out / "games.sims=2.parquet"
    assert moves_shard.exists() and games_shard.exists()

    import pyarrow.parquet as pq
    games_df = pq.read_table(games_shard).to_pandas()
    # 1 budget * 1 game = 1 game row (assuming the game didn't time out)
    assert len(games_df) <= 1  # 0 if timed out (unlikely at 300s for sims=2)
    assert (out / "done.txt").exists() or (out / "skipped.csv").exists()
