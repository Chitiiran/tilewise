from pathlib import Path

from catan_mcts.experiments.e2_ucb_c_sweep import main


def test_e2_smoke_run(tmp_path: Path):
    # Smoke ONLY — see e1 test for rationale on tiny params.
    # v2: per-c checkpoint flush; data lands in games.c=*.parquet shards.
    out = main(
        out_root=tmp_path, num_games=1, c_grid=[0.5, 4.0], sims=2, seed_base=2000,
    )
    # 2 c-values × 1 game → 2 game rows total across 2 shards.
    import pyarrow.parquet as pq
    shards = sorted(out.glob("games.c=*.parquet"))
    assert len(shards) == 2, f"expected 2 c-value shards, got {[s.name for s in shards]}"
    total_games = sum(len(pq.read_table(s).to_pandas()) for s in shards)
    # Some games may time out under the 300s cap on slow hardware; allow 1-2.
    assert 0 < total_games <= 2
