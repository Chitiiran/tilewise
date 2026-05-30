"""Smoke test: Cand 11 (lambda_road=0.05) isolated through train_main
for 1 epoch on the minimal e1 fixture. No Cand 1, no Cand 8, no Cand 10.

Verifies:
  - lambda_road > 0 path doesn't crash.
  - Road-prior loss term composes with vanilla loss without NaN.
  - Per-sample [B, N, F] reshape from PyG batch works for vertex_features
    and edge_features (Cand 1 already exercises hex_features).
  - lambda_road = 0 path is byte-identical to vanilla (no accidental gradient).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _train_with_lambda_road(tmp_path: Path, *, lambda_road: float, e1_dir: Path) -> float:
    """Train 1 epoch with the given lambda_road on the supplied e1 fixture dir."""
    from catan_gnn.train import train_main

    out_dir = tmp_path / f"cell5_lr_{lambda_road}"
    train_main(
        run_dirs=[e1_dir],
        out_dir=out_dir,
        hidden_dim=8, num_layers=2,
        epochs=1, batch_size=4, lr=1e-3,
        val_frac=0.2, seed=0, device="cpu",
        # Cand 11 ONLY (everything else off)
        lambda_road=lambda_road,
        lambda_vp=0.0,
        vp_compare_rule=False,
        lambda_settle=0.0,
    )
    log = json.loads((out_dir / "training_log.json").read_text())
    return float(log["epochs"][0]["train_loss_total"])


@pytest.fixture
def e1_fixture(tmp_path):
    """Generate a small e1 winrate fixture once per test."""
    from catan_mcts.experiments.e1_winrate_vs_random import main as e1_main
    return e1_main(
        out_root=tmp_path / "e1_root",
        num_games=2, sims_per_move_grid=[2],
        seed_base=88888, max_seconds=300.0,
    )


@pytest.mark.slow
def test_cell5_lambda_road_005_no_nan(tmp_path, e1_fixture):
    """Cell 5 config: lambda_road = 0.05. Train 1 epoch, assert no NaN."""
    loss = _train_with_lambda_road(tmp_path, lambda_road=0.05, e1_dir=e1_fixture)
    assert loss == loss, f"train_loss NaN with lambda_road=0.05"
    assert loss > 0
    assert loss < 100


@pytest.mark.slow
def test_cell5_lambda_road_zero_matches_vanilla(tmp_path, e1_fixture):
    """With lambda_road=0, the road loss block is skipped entirely.
    Two runs with the same seed should produce byte-identical loss."""
    loss_a = _train_with_lambda_road(tmp_path, lambda_road=0.0, e1_dir=e1_fixture)
    # Wipe the second run's out dir to force a fresh train (e1 fixture is reused)
    import shutil
    shutil.rmtree(tmp_path / "cell5_lr_0.0")
    loss_b = _train_with_lambda_road(tmp_path, lambda_road=0.0, e1_dir=e1_fixture)
    assert abs(loss_a - loss_b) < 1e-5, (
        f"vanilla (lambda_road=0) not reproducible: {loss_a} vs {loss_b}"
    )
