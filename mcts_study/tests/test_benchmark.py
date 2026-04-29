"""Tests for catan_gnn.benchmark: bench-2 produces a JSON with bench2_value_mae + bench2_policy_kl."""
import json
from pathlib import Path

import torch

from catan_gnn.benchmark import bench2_main


def _make_minimal_run(tmp_path: Path):
    from catan_mcts.experiments.e1_winrate_vs_random import main
    return main(
        out_root=tmp_path, num_games=2,
        sims_per_move_grid=[2], seed_base=22222,
        max_seconds=300.0,
    )


def test_bench2_writes_json(tmp_path: Path):
    from catan_gnn.train import train_main

    run_dir = _make_minimal_run(tmp_path / "run")
    out_dir = tmp_path / "gnn_v0"
    train_main(
        run_dirs=[run_dir], out_dir=out_dir,
        hidden_dim=32, num_layers=2,
        epochs=1, batch_size=4, lr=1e-3,
        w_value=1.0, w_policy=1.0, seed=0,
    )
    bench2_main(
        checkpoint=out_dir / "checkpoint.pt",
        run_dirs=[run_dir],
        out_path=out_dir / "bench2.json",
        n_positions=10,
        lookahead_depth=3,  # tiny for tests
    )
    j = json.loads((out_dir / "bench2.json").read_text())
    assert "bench2_value_mae" in j
    assert "bench2_policy_kl" in j
    assert j["n_positions"] == 10
