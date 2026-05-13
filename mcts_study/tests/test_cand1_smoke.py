"""Smoke test: Cand 1 (lambda_settle=0.20) isolated through train_main
for 1 epoch on the minimal e1 fixture. No Cand 8, no Cand 10.

Verifies:
  - lambda_settle > 0 path doesn't crash
  - Settlement-prior loss term composes with vanilla loss without NaN
  - hex_features reshape from PyG batch works correctly
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.mark.slow
def test_cand1_isolated_runs_one_epoch(tmp_path):
    """Train tiny GNN for 1 epoch with ONLY Cand 1 active (Cell 2 config)."""
    from catan_mcts.experiments.e1_winrate_vs_random import main as e1_main
    from catan_gnn.train import train_main

    e1_out = e1_main(
        out_root=tmp_path / "e1",
        num_games=2, sims_per_move_grid=[2],
        seed_base=88888, max_seconds=300.0,
    )

    out_dir = tmp_path / "cell2_smoke"
    train_main(
        run_dirs=[e1_out],
        out_dir=out_dir,
        hidden_dim=8, num_layers=2,
        epochs=1, batch_size=4, lr=1e-3,
        val_frac=0.2, seed=0, device="cpu",
        # Cand 1 ONLY
        lambda_settle=0.20,
        # Cand 8 and Cand 10 OFF
        lambda_vp=0.0,
        vp_compare_rule=False,
    )

    assert (out_dir / "checkpoint.pt").exists()
    log = json.loads((out_dir / "training_log.json").read_text())
    assert "epochs" in log
    assert len(log["epochs"]) == 1
    ep0 = log["epochs"][0]
    assert ep0["train_loss_total"] == ep0["train_loss_total"], "train_loss NaN"
    assert ep0["val_loss_total"] == ep0["val_loss_total"], "val_loss NaN"
    assert ep0["train_loss_total"] < 100
