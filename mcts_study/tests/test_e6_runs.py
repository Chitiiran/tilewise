"""Smoke test for e6_mcts_gnn_winrate."""
from pathlib import Path

import torch

from catan_mcts.experiments.e6_mcts_gnn_winrate import main


def _train_tiny_gnn(tmp_path: Path) -> Path:
    from catan_mcts.experiments.e1_winrate_vs_random import main as e1_main
    from catan_gnn.train import train_main

    run_dir = e1_main(
        out_root=tmp_path / "e1run", num_games=2,
        sims_per_move_grid=[2], seed_base=33333, max_seconds=300.0,
    )
    out_dir = tmp_path / "gnn_v0"
    train_main(
        run_dirs=[run_dir], out_dir=out_dir,
        hidden_dim=32, num_layers=2, epochs=1, batch_size=4,
        lr=1e-3, w_value=1.0, w_policy=1.0, seed=0,
    )
    return out_dir / "checkpoint.pt"


def test_e6_smoke_run(tmp_path: Path):
    ckpt = _train_tiny_gnn(tmp_path / "train")
    out = main(
        out_root=tmp_path / "e6",
        checkpoint=ckpt,
        num_games=1, sims_grid=[2],
        seed_base=44444, max_seconds=300.0,
    )
    games_shard = out / "games.sims=2.parquet"
    assert games_shard.exists()
    assert (out / "done.txt").exists() or (out / "skipped.csv").exists()
