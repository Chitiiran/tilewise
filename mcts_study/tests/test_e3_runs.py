from pathlib import Path

from catan_mcts.experiments.e3_rollout_policy import main


def test_e3_smoke_run(tmp_path: Path):
    # Smoke ONLY — see e1 test for sizing rationale.
    # v2: per-(policy,sims)-cell checkpoint; data lands in games.<policy>-sims=N.parquet.
    out = main(out_root=tmp_path, num_games=1, sims_grid=[2], seed_base=3000)
    import pyarrow.parquet as pq
    shards = sorted(out.glob("games.*-sims=*.parquet"))
    # 2 rollout policies × 1 sim budget = 2 shards.
    assert len(shards) == 2, f"expected 2 shards, got {[s.name for s in shards]}"
    total_games = sum(len(pq.read_table(s).to_pandas()) for s in shards)
    assert 0 < total_games <= 2
