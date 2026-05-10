"""Tests for catan_gnn.mid_training_tournament — the per-N-epoch
tournament eval hook used by Phase 1 of the loss-augmentation roadmap.

Three surfaces are tested:

1. **summarize_tournament_run_dir**: pure parquet-parser. Given an
   existing e10_v3_tournament output dir (worker*/games.rot=*.parquet
   layout), compute PureGnn winrate, GnnMcts winrate, etc. Tested
   against a known fixture from grid_pass3_lastepoch/h32_l2 which has
   exactly 120 games with verified winner counts.

2. **should_stop_for_drop**: pure decision rule. Given the history of
   mid-training tournament PureGnn winrates and a drop threshold,
   return whether to early-stop. Encodes the plan's gating logic:
   "stop when winrate drops by >= 3 games (2.5pp) compared to the
   previous mid-training tournament; epoch-5 baseline = 0% so first
   tournament always passes; rule first fires at epoch 10."

3. (integration with train.py is covered by the smoke-train test, not
   here — that's wired in via train.py's existing test surface.)
"""
from __future__ import annotations

from pathlib import Path

import pytest

from catan_gnn.mid_training_tournament import (
    should_stop_for_drop,
    summarize_tournament_run_dir,
)


# Fixture: 120-game tournament from pass-3 lastepoch h32_l2 cell.
# Independently verified above: PureGnn=7, LookaheadV3=110, GnnMcts=2,
# Random=1. PureGnn winrate = 7/120 = 0.0583...
FIXTURE_DIR = Path(
    "runs/v3/grid_pass3_lastepoch/h32_l2/"
    "2026-05-04T21-53-e10_v3_tournament"
)
FIXTURE_TOTAL_GAMES = 120
FIXTURE_PUREGNN_WINS = 7
FIXTURE_GNNMCTS_WINS = 2
FIXTURE_LOOKAHEAD_WINS = 110
FIXTURE_RANDOM_WINS = 1


# === summarize_tournament_run_dir ===

def test_summarize_returns_expected_winrate():
    if not FIXTURE_DIR.exists():
        pytest.skip(f"Tournament fixture not present at {FIXTURE_DIR}")
    s = summarize_tournament_run_dir(FIXTURE_DIR)
    assert s["total_games"] == FIXTURE_TOTAL_GAMES
    assert s["pure_gnn_wins"] == FIXTURE_PUREGNN_WINS
    # Winrate is computed exactly from wins / total.
    expected = FIXTURE_PUREGNN_WINS / FIXTURE_TOTAL_GAMES
    assert s["pure_gnn_winrate"] == pytest.approx(expected, abs=1e-9)


def test_summarize_returns_all_role_counts():
    if not FIXTURE_DIR.exists():
        pytest.skip(f"Tournament fixture not present at {FIXTURE_DIR}")
    s = summarize_tournament_run_dir(FIXTURE_DIR)
    assert s["pure_gnn_wins"] == FIXTURE_PUREGNN_WINS
    assert s["gnn_mcts_wins"] == FIXTURE_GNNMCTS_WINS
    assert s["lookahead_v3_wins"] == FIXTURE_LOOKAHEAD_WINS
    assert s["random_wins"] == FIXTURE_RANDOM_WINS
    # All wins + no_winner_games should equal total_games.
    assigned = (s["pure_gnn_wins"] + s["gnn_mcts_wins"]
                + s["lookahead_v3_wins"] + s["random_wins"])
    assert assigned + s["no_winner_games"] == s["total_games"]


def test_summarize_raises_on_missing_parquet(tmp_path):
    """If run dir has no parquet files, raise rather than return 0% winrate
    (silent zero would be ambiguous: "no games run" vs "ran but PureGnn
    lost all of them")."""
    empty = tmp_path / "empty_run"
    empty.mkdir()
    with pytest.raises((FileNotFoundError, RuntimeError, ValueError)):
        summarize_tournament_run_dir(empty)


# === should_stop_for_drop decision rule ===

def test_first_tournament_never_stops():
    """At epoch 5, no prior tournament exists; rule shouldn't fire."""
    assert should_stop_for_drop(
        history=[],
        current_wins=10,
        total_games=120,
        drop_threshold=3,
    ) is False


def test_strict_improvement_continues():
    """Winrate went up — always continue."""
    assert should_stop_for_drop(
        history=[8],
        current_wins=15,
        total_games=120,
        drop_threshold=3,
    ) is False


def test_equal_winrate_continues():
    """Per plan: 'stop when winrate drops by >= 3 games'. Equal is not
    a drop, so continue."""
    assert should_stop_for_drop(
        history=[15],
        current_wins=15,
        total_games=120,
        drop_threshold=3,
    ) is False


def test_small_drop_within_threshold_continues():
    """Drop of 2 games is below the 3-game threshold."""
    assert should_stop_for_drop(
        history=[15],
        current_wins=13,
        total_games=120,
        drop_threshold=3,
    ) is False


def test_drop_at_threshold_stops():
    """Drop exactly equal to threshold triggers stop."""
    assert should_stop_for_drop(
        history=[15],
        current_wins=12,
        total_games=120,
        drop_threshold=3,
    ) is True


def test_drop_above_threshold_stops():
    """Bigger drop also stops."""
    assert should_stop_for_drop(
        history=[15],
        current_wins=5,
        total_games=120,
        drop_threshold=3,
    ) is True


def test_compares_to_previous_not_best():
    """The rule compares to the IMMEDIATELY-PRECEDING tournament, not
    to the running max. That's intentional: we want to catch a fresh
    regression even if the run has been climbing."""
    # History: 5 -> 12 -> 18 -> now 16 (drop of 2 from 18; below threshold)
    assert should_stop_for_drop(
        history=[5, 12, 18],
        current_wins=16,
        total_games=120,
        drop_threshold=3,
    ) is False
    # History: 5 -> 12 -> 18 -> now 14 (drop of 4 from 18; at threshold)
    assert should_stop_for_drop(
        history=[5, 12, 18],
        current_wins=14,
        total_games=120,
        drop_threshold=3,
    ) is True


# === End-to-end smoke ===

EXISTING_CHECKPOINT = Path("runs/v3/grid/training_h32_l2/checkpoint_best.pt")


@pytest.mark.slow
def test_run_mid_training_tournament_end_to_end(tmp_path):
    """Call run_mid_training_tournament against an existing h32_l2
    checkpoint with a 4-game configuration (1 game/seating × 4 rotations).
    Proves the helper:
      - constructs the right e10 call
      - the tournament runs to completion
      - the output dir contains parseable parquets
      - summarize_tournament_run_dir picks them up

    This is slow because each game takes 5-30s of MCTS play."""
    if not EXISTING_CHECKPOINT.exists():
        pytest.skip(f"Smoke checkpoint missing at {EXISTING_CHECKPOINT}")

    from catan_gnn.mid_training_tournament import run_mid_training_tournament

    out_root = tmp_path / "smoke_mt_run"
    result = run_mid_training_tournament(
        epoch=999,
        checkpoint_path=EXISTING_CHECKPOINT,
        out_root=out_root,
        hidden_dim=32, num_layers=2,
        num_games_per_seating=1,  # 1 × 4 rotations = 4 games
        sims=10,                   # tiny for smoke (real runs use 100)
        lookahead_depth=2,
        base_sims_v3=50,           # SIM_FLOOR=50 per players_v3.py
        seed_base=30_000_000,
        max_seconds=120.0,
        device="cpu",
    )
    s = result.summary
    assert s["total_games"] == 4
    assigned = (s["pure_gnn_wins"] + s["gnn_mcts_wins"]
                + s["lookahead_v3_wins"] + s["random_wins"])
    assert assigned + s["no_winner_games"] == 4
    assert 0.0 <= s["pure_gnn_winrate"] <= 1.0
    assert result.epoch == 999
    assert result.run_dir.exists()
