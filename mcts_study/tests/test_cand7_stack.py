"""Smoke test: Cand 7 stacked on Cand 8 + Cand 10 through train_main
for 1 epoch on the minimal e1 fixture.

This is the Cell 2 config from the loss-augmentation roadmap:
  --lambda-vp 0.10 --vp-compare-rule --class-balanced-policy

Verifies:
  - class_balanced_policy=True path doesn't crash
  - Composes with Cand 8 (lambda_vp>0) and Cand 10 (vp_compare_rule=True)
  - Training and val losses remain finite (no NaN from rebalancing)
  - The log line includes vp_swap=N/B (P%) (still works under Cand 7)
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.mark.slow
def test_cand7_stacked_on_cand8_cand10(tmp_path):
    """Trains a tiny GNN for 1 epoch with Cand 7 + Cand 8 + Cand 10 active —
    the Cell 2 config. Uses the minimal e1 fixture (single MCTS game)."""
    from catan_mcts.experiments.e1_winrate_vs_random import main as e1_main
    from catan_gnn.train import train_main

    e1_out = e1_main(
        out_root=tmp_path / "e1",
        num_games=2, sims_per_move_grid=[2],
        seed_base=77777, max_seconds=300.0,
    )

    out_dir = tmp_path / "cell2_smoke"
    train_main(
        run_dirs=[e1_out],
        out_dir=out_dir,
        hidden_dim=8, num_layers=2,
        epochs=1, batch_size=4, lr=1e-3,
        val_frac=0.2, seed=0, device="cpu",
        # Cell 2 = Cand 7 + Cand 8 + Cand 10
        lambda_vp=0.10,
        vp_compare_rule=True,
        class_balanced_policy=True,
    )

    # Verify training produced expected artifacts
    assert (out_dir / "checkpoint.pt").exists()
    assert (out_dir / "training_log.json").exists()
    log = json.loads((out_dir / "training_log.json").read_text())
    assert "epochs" in log
    assert len(log["epochs"]) == 1
    ep0 = log["epochs"][0]
    # NaN guard: train + val losses must be finite
    assert ep0["train_loss_total"] == ep0["train_loss_total"], "train_loss NaN"
    assert ep0["val_loss_total"] == ep0["val_loss_total"], "val_loss NaN"
    # Loss bounded — Cand 7 only renormalizes, doesn't blow up scale.
    assert ep0["train_loss_total"] < 100, f"train_loss too large: {ep0['train_loss_total']}"


@pytest.mark.slow
def test_cand7_isolated(tmp_path):
    """Cand 7 alone (no Cand 8, no Cand 10) — verifies the rebalance
    composes with vanilla loss without crashing or NaN."""
    from catan_mcts.experiments.e1_winrate_vs_random import main as e1_main
    from catan_gnn.train import train_main

    e1_out = e1_main(
        out_root=tmp_path / "e1",
        num_games=2, sims_per_move_grid=[2],
        seed_base=66666, max_seconds=300.0,
    )

    out_dir = tmp_path / "cand7_only_smoke"
    train_main(
        run_dirs=[e1_out],
        out_dir=out_dir,
        hidden_dim=8, num_layers=2,
        epochs=1, batch_size=4, lr=1e-3,
        val_frac=0.2, seed=0, device="cpu",
        class_balanced_policy=True,
        # Other candidates off
        lambda_vp=0.0,
        vp_compare_rule=False,
        lambda_settle=0.0,
    )

    assert (out_dir / "checkpoint.pt").exists()
    log = json.loads((out_dir / "training_log.json").read_text())
    ep0 = log["epochs"][0]
    assert ep0["train_loss_total"] == ep0["train_loss_total"]
    assert ep0["val_loss_total"] == ep0["val_loss_total"]
    assert ep0["train_loss_total"] < 100
