from pathlib import Path

from catan_mcts.experiments.e5_lookahead_depth import main


def test_e5_smoke_run(tmp_path: Path):
    # Smoke ONLY — exercises the e5 lookahead-depth experiment end-to-end with
    # the smallest possible grid (1 depth × 1 sim cell × 1 game).
    out = main(
        out_root=tmp_path,
        num_games=1,
        depth_grid=[3],
        sims_grid=[2],
        seed_base=5000,
        max_seconds=300.0,
    )
    moves_shard = out / "moves.depth=3-sims=2.parquet"
    games_shard = out / "games.depth=3-sims=2.parquet"
    assert moves_shard.exists() and games_shard.exists()

    import pyarrow.parquet as pq
    games_df = pq.read_table(games_shard).to_pandas()
    assert len(games_df) <= 1
    assert (out / "done.txt").exists() or (out / "skipped.csv").exists()


def test_e5_parallel_smoke(tmp_path: Path):
    out = main(
        out_root=tmp_path,
        num_games=2,
        depth_grid=[3],
        sims_grid=[2],
        seed_base=6000,
        max_seconds=300.0,
        workers=2,
    )
    assert (out / "worker0").is_dir()
    assert (out / "worker1").is_dir()
    for w in (0, 1):
        wd = out / f"worker{w}"
        assert (wd / "done.txt").exists() or (wd / "skipped.csv").exists()
