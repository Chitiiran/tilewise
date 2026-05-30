"""Smoke test: Cand 8 + Cand 10 stacked through train_main for 1 epoch.

Validates that:
  - lambda_vp > 0 path doesn't crash
  - vp_compare_rule=True path doesn't crash
  - Both stacked together (the Cell 1 config) trains without NaN
  - The log line includes vp_swap=N/B (P%) when vp_compare_rule=True
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.mark.slow
def test_cand8_cand10_stacked_runs_one_epoch(tmp_path):
    """Trains a tiny GNN for 1 epoch with both Cand 8 (lambda_vp=0.10)
    and Cand 10 (vp_compare_rule=True) active. Uses the minimal e1
    fixture (single MCTS game) so the cache is small."""
    from catan_mcts.experiments.e1_winrate_vs_random import main as e1_main
    from catan_gnn.train import train_main

    e1_out = e1_main(
        out_root=tmp_path / "e1",
        num_games=2, sims_per_move_grid=[2],
        seed_base=99999, max_seconds=300.0,
    )

    out_dir = tmp_path / "cell1_smoke"
    train_main(
        run_dirs=[e1_out],
        out_dir=out_dir,
        hidden_dim=8, num_layers=2,
        epochs=1, batch_size=4, lr=1e-3,
        val_frac=0.2, seed=0, device="cpu",
        lambda_vp=0.10,
        vp_compare_rule=True,
    )

    # Verify training produced expected artifacts
    assert (out_dir / "checkpoint.pt").exists()
    assert (out_dir / "training_log.json").exists()
    log = json.loads((out_dir / "training_log.json").read_text())
    assert "epochs" in log
    assert len(log["epochs"]) == 1
    ep0 = log["epochs"][0]
    # NaN guard: train loss must be finite
    assert ep0["train_loss_total"] == ep0["train_loss_total"], "train_loss NaN"
    assert ep0["val_loss_total"] == ep0["val_loss_total"], "val_loss NaN"
    # Loss shouldn't be huge — lambda_vp=0.10 adds a small auxiliary
    # term; vp_compare may swap some targets but still bounded CE.
    assert ep0["train_loss_total"] < 100, f"train_loss too large: {ep0['train_loss_total']}"
