"""Tests for catan_gnn.analysis.trade_value.classify_trades_in_game.

Phase 0 of the loss-augmentation roadmap. The classifier replays the v3
engine through a recorded action_history, intercepts every ProposeTrade
(action IDs 260..279), and reports whether each trade was accepted (hand
delta on proposer) and value-adding (followed by Build/Buy before next
EndTurn by the same proposer).

Engine-side facts these tests rely on (cited):
  - catan_engine/src/rules.rs:347-374 — ProposeTrade resolution: iterate
    opponents in seat order from current_player + 1; first with >=1 of
    `get` accepts a 1-for-1 swap. If none, silent no-op.
  - catan_engine/src/actions.rs:49-58 — action ID layout. ProposeTrade
    occupies 260..279, EndTurn=204, RollDice=205, BuildSettlement=0..53,
    BuildCity=54..107, BuildRoad=108..179, BuyDevCard=226.
  - catan_engine/src/lib.rs:140-141 — Engine.clone() exposed via PyO3.

Fixture game: seed=21100000 from worker0 of the 100k corpus
(runs/v3/2026-05-05T05-50-e9_v3_data_gen_100k_w12/.../worker0/
games.v3-final.parquet). Inspected to have 51 ProposeTrade actions in
its 241-step history; first trade at history index 20 (action_id=276).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from catan_gnn.analysis.trade_value import classify_trades_in_game


CORPUS_GAMES = Path(
    "runs/v3/2026-05-05T05-50-e9_v3_data_gen_100k_w12/"
    "2026-05-05T09-50-e9_v3_data_gen/worker0/games.v3-final.parquet"
)


@pytest.fixture(scope="module")
def fixture_game():
    """Return (seed, action_history) of the first game in worker0."""
    if not CORPUS_GAMES.exists():
        pytest.skip(f"100k corpus not present at {CORPUS_GAMES}")
    df = pq.read_table(str(CORPUS_GAMES)).to_pandas()
    row = df.iloc[0]
    return int(row["seed"]), [int(a) for a in row["action_history"]]


def test_classifier_returns_empty_for_no_trades_history(fixture_game):
    """A game replayed only through its setup phase has no ProposeTrade
    actions (cited: 100k corpus mi=0..3 are 100% Settle/Road, never
    ProposeTrade — see scratch_check_setup_samples.py output in
    2026-05-09 loss-aug doc, lines 313-329).

    Truncate the fixture's history to the first 16 actions (=4 setup
    settlements + 4 setup roads + chance interleaves). Verify classifier
    returns []."""
    seed, ah = fixture_game
    # Take first 16 raw entries; that's strictly inside setup phase.
    setup_only = ah[:16]
    # Sanity: none of these should be a ProposeTrade
    propose_ids_in_prefix = [
        a for a in setup_only
        if not (a & 0x80000000) and 260 <= a <= 279
    ]
    assert propose_ids_in_prefix == [], (
        f"Test premise violated: setup prefix has ProposeTrades: "
        f"{propose_ids_in_prefix}"
    )
    result = classify_trades_in_game(seed, setup_only)
    assert result == []


def test_classifier_count_matches_history(fixture_game):
    """Classifier must return one record per ProposeTrade action in history."""
    seed, ah = fixture_game
    expected_count = sum(
        1 for a in ah
        if not (a & 0x80000000) and 260 <= a <= 279
    )
    assert expected_count > 0, "Fixture chosen poorly: no trades in game"
    result = classify_trades_in_game(seed, ah)
    assert len(result) == expected_count


def test_classifier_records_have_required_fields(fixture_game):
    """Each record must have keys: proposer (0..3), action_id (260..279),
    accepted (bool), value_adding (bool)."""
    seed, ah = fixture_game
    result = classify_trades_in_game(seed, ah)
    assert len(result) > 0
    rec = result[0]
    assert set(rec.keys()) >= {"proposer", "action_id", "accepted", "value_adding"}
    assert 0 <= rec["proposer"] <= 3
    assert 260 <= rec["action_id"] <= 279
    assert isinstance(rec["accepted"], bool)
    assert isinstance(rec["value_adding"], bool)


def test_classifier_value_adding_implies_accepted(fixture_game):
    """value_adding=True must imply accepted=True (per the design doc:
    'Trade was value-adding iff (accepted) AND ...')."""
    seed, ah = fixture_game
    result = classify_trades_in_game(seed, ah)
    for rec in result:
        if rec["value_adding"]:
            assert rec["accepted"], (
                f"Inconsistent record: value_adding=True but accepted=False: {rec}"
            )


def test_classifier_first_trade_accepted_flag_matches_independent_replay(fixture_game):
    """For the first ProposeTrade in the fixture game, replay the engine
    independently up to that action, snapshot proposer's hand, step,
    snapshot again, and check accepted flag matches the classifier."""
    seed, ah = fixture_game
    from catan_bot import _engine

    eng = _engine.Engine.with_rules(seed, 5, False)
    first_trade_idx = None
    for i, a in enumerate(ah):
        if not (a & 0x80000000) and 260 <= a <= 279:
            first_trade_idx = i
            break
    assert first_trade_idx is not None

    # Replay up to (not including) the first trade
    for j in range(first_trade_idx):
        a = ah[j]
        if a & 0x80000000:
            eng.apply_chance_outcome(a & 0x7FFFFFFF)
        else:
            eng.step(a)

    proposer = int(eng.current_player())
    hand_before = eng.all_hands()[proposer].copy()
    eng.step(int(ah[first_trade_idx]))
    hand_after = eng.all_hands()[proposer].copy()
    expected_accepted = bool(np.any(hand_before != hand_after))

    # Now run the classifier on the full history; check first record matches
    result = classify_trades_in_game(seed, ah)
    assert result[0]["proposer"] == proposer
    assert result[0]["action_id"] == int(ah[first_trade_idx])
    assert result[0]["accepted"] == expected_accepted
