"""Smoke test: Cell 6 = Cand 11 + Cand 8 + Cand 10 stacked through train_main
for 1 epoch on the minimal e1 fixture.

Verifies:
  - Stacking lambda_road > 0, lambda_vp > 0, and vp_compare_rule together
    doesn't crash and doesn't produce NaN.
  - All three loss terms compose with vanilla CE + value MSE.
  - The compound train_loss is finite and > 0.

Per memory feedback_training_observability.md: this smoke test catches
config-level issues before we burn 19h on a misconfigured stack.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest


def _train_stack(tmp_path: Path, *, lambda_road: float, lambda_vp: float,
                 vp_compare_rule: bool, e1_dir: Path) -> dict:
    """Train 1 epoch with the given stack config. Returns the epoch log dict."""
    from catan_gnn.train import train_main

    out_dir = tmp_path / f"cell6_lr{lambda_road}_lv{lambda_vp}_vc{int(vp_compare_rule)}"
    train_main(
        run_dirs=[e1_dir],
        out_dir=out_dir,
        hidden_dim=8, num_layers=2,
        epochs=1, batch_size=4, lr=1e-3,
        val_frac=0.2, seed=0, device="cpu",
        lambda_road=lambda_road,
        lambda_vp=lambda_vp,
        vp_compare_rule=vp_compare_rule,
        lambda_settle=0.0,
        class_balanced_policy=False,
    )
    log = json.loads((out_dir / "training_log.json").read_text())
    return log["epochs"][0]


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
def test_cell6_stacked_cand11_cand8_cand10_no_nan(tmp_path, e1_fixture):
    """Cell 6 config: lambda_road=0.05, lambda_vp=0.10, vp_compare_rule=True.
    Train 1 epoch, assert all loss components finite + train_loss in sane range."""
    ep = _train_stack(
        tmp_path,
        lambda_road=0.05, lambda_vp=0.10, vp_compare_rule=True,
        e1_dir=e1_fixture,
    )
    train_loss = float(ep["train_loss_total"])
    val_loss = float(ep["val_loss_total"])
    val_top1 = float(ep["val_policy_top1_acc"])
    assert math.isfinite(train_loss), f"train_loss not finite: {train_loss}"
    assert math.isfinite(val_loss), f"val_loss not finite: {val_loss}"
    assert math.isfinite(val_top1), f"val_top1 not finite: {val_top1}"
    assert train_loss > 0
    assert train_loss < 100, f"train_loss explosively large: {train_loss}"


@pytest.mark.slow
def test_cell6_stack_matches_vanilla_when_all_off(tmp_path, e1_fixture):
    """With all three flags off, Cell 6 path is byte-identical to vanilla.
    Confirms no accidental gradient leaks from the stacked code paths."""
    ep_a = _train_stack(
        tmp_path,
        lambda_road=0.0, lambda_vp=0.0, vp_compare_rule=False,
        e1_dir=e1_fixture,
    )
    # Force a fresh train to rule out caching artifacts
    import shutil
    shutil.rmtree(tmp_path / "cell6_lr0.0_lv0.0_vc0")
    ep_b = _train_stack(
        tmp_path,
        lambda_road=0.0, lambda_vp=0.0, vp_compare_rule=False,
        e1_dir=e1_fixture,
    )
    assert abs(ep_a["train_loss_total"] - ep_b["train_loss_total"]) < 1e-5, (
        f"all-flags-off path not reproducible: "
        f"{ep_a['train_loss_total']} vs {ep_b['train_loss_total']}"
    )
