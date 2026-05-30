"""Tests for Cand 8 (Action-class VP prior) of the loss-augmentation roadmap.

The VP prior is a per-sample policy target that puts equal probability
mass on all LEGAL actions whose action_class is in
{BuildSettlement, BuildCity, PlayVpCard, BuyDevCard} — the actions that
yield VP (directly, or with non-zero probability via dev card draw).
All other legal actions get zero target probability.

If no VP-yielding action is legal in a state, the prior degrades to a
uniform distribution over the legal actions (no signal).

Used as the target distribution in an auxiliary KL term:
  loss += lambda_vp * KL(softmax(logits) || vp_target)

Engine action ID layout (cited catan_engine/src/actions.rs:121-127, 49-58):
  BuildSettlement:   0..53
  BuildCity:        54..107
  BuildRoad:       108..179
  MoveRobber:      180..198
  Discard:         199..203
  EndTurn:             204
  RollDice:            205
  TradeBank:       206..225
  BuyDevCard:          226
  PlayKnight:          227
  PlayRoadBuilding:    228
  PlayMonopoly:    229..233
  PlayYearOfPlenty:234..258
  PlayVpCard:          259
  ProposeTrade:    260..279
"""
from __future__ import annotations

import numpy as np
import torch
import pytest

from catan_gnn.action_classes import (
    ACTION_CLASS,
    CLASS_VP_VALUE,
    ActionClass,
    build_vp_prior_target,
)


# ============ ACTION_CLASS table ============

def test_action_class_size():
    """Table covers full 280-action space (cited ACTION_SPACE_SIZE=280)."""
    assert len(ACTION_CLASS) == 280


def test_action_class_boundaries():
    """Spot-check the boundaries of each class (cited actions.rs:121-127)."""
    # BuildSettlement: 0..53
    assert ACTION_CLASS[0] == ActionClass.BUILD_SETTLEMENT
    assert ACTION_CLASS[53] == ActionClass.BUILD_SETTLEMENT
    # BuildCity: 54..107
    assert ACTION_CLASS[54] == ActionClass.BUILD_CITY
    assert ACTION_CLASS[107] == ActionClass.BUILD_CITY
    # BuildRoad: 108..179
    assert ACTION_CLASS[108] == ActionClass.BUILD_ROAD
    assert ACTION_CLASS[179] == ActionClass.BUILD_ROAD
    # MoveRobber: 180..198
    assert ACTION_CLASS[180] == ActionClass.MOVE_ROBBER
    assert ACTION_CLASS[198] == ActionClass.MOVE_ROBBER
    # Discard: 199..203
    assert ACTION_CLASS[199] == ActionClass.DISCARD
    assert ACTION_CLASS[203] == ActionClass.DISCARD
    # EndTurn: 204
    assert ACTION_CLASS[204] == ActionClass.END_TURN
    # RollDice: 205
    assert ACTION_CLASS[205] == ActionClass.ROLL_DICE
    # TradeBank: 206..225
    assert ACTION_CLASS[206] == ActionClass.TRADE_BANK
    assert ACTION_CLASS[225] == ActionClass.TRADE_BANK
    # BuyDevCard: 226
    assert ACTION_CLASS[226] == ActionClass.BUY_DEV_CARD
    # PlayKnight: 227
    assert ACTION_CLASS[227] == ActionClass.PLAY_KNIGHT
    # PlayRoadBuilding: 228
    assert ACTION_CLASS[228] == ActionClass.PLAY_ROAD_BUILDING
    # PlayMonopoly: 229..233
    assert ACTION_CLASS[229] == ActionClass.PLAY_MONOPOLY
    assert ACTION_CLASS[233] == ActionClass.PLAY_MONOPOLY
    # PlayYearOfPlenty: 234..258
    assert ACTION_CLASS[234] == ActionClass.PLAY_YEAR_OF_PLENTY
    assert ACTION_CLASS[258] == ActionClass.PLAY_YEAR_OF_PLENTY
    # PlayVpCard: 259
    assert ACTION_CLASS[259] == ActionClass.PLAY_VP_CARD
    # ProposeTrade: 260..279
    assert ACTION_CLASS[260] == ActionClass.PROPOSE_TRADE
    assert ACTION_CLASS[279] == ActionClass.PROPOSE_TRADE


# ============ CLASS_VP_VALUE table ============

def test_vp_yielding_classes_value_one():
    """Direct +1 VP classes (cited rules.rs:181, 210, 266)."""
    assert CLASS_VP_VALUE[ActionClass.BUILD_SETTLEMENT] == 1.0
    assert CLASS_VP_VALUE[ActionClass.BUILD_CITY] == 1.0
    assert CLASS_VP_VALUE[ActionClass.PLAY_VP_CARD] == 1.0


def test_buy_dev_card_probabilistic_value():
    """BuyDevCard has expected VP via VP-card draw probability."""
    v = CLASS_VP_VALUE[ActionClass.BUY_DEV_CARD]
    # Per plan: ~1/14 chance of drawing VP card. Exact value depends on
    # deck composition; we just require it's in (0, 1).
    assert 0.0 < v < 1.0


def test_non_vp_classes_zero():
    """Roads / trades / EndTurn / etc. have zero VP weight."""
    for c in [
        ActionClass.BUILD_ROAD,
        ActionClass.TRADE_BANK,
        ActionClass.PROPOSE_TRADE,
        ActionClass.END_TURN,
        ActionClass.ROLL_DICE,
        ActionClass.MOVE_ROBBER,
        ActionClass.DISCARD,
        ActionClass.PLAY_KNIGHT,
        ActionClass.PLAY_ROAD_BUILDING,
        ActionClass.PLAY_MONOPOLY,
        ActionClass.PLAY_YEAR_OF_PLENTY,
    ]:
        assert CLASS_VP_VALUE[c] == 0.0, f"{c} should be 0.0, got {CLASS_VP_VALUE[c]}"


# ============ build_vp_prior_target ============

def test_target_uniform_over_vp_actions_when_legal():
    """When BuildCity and BuildSettlement are legal, target puts equal
    mass on each (proportional to CLASS_VP_VALUE, both 1.0)."""
    legal = torch.zeros(280, dtype=torch.bool)
    legal[5] = True       # BuildSettlement
    legal[60] = True      # BuildCity
    legal[150] = True     # BuildRoad (no VP)
    legal[204] = True     # EndTurn (no VP)
    target = build_vp_prior_target(legal)
    assert target.shape == (280,)
    # Sum = 1.0
    assert target.sum().item() == pytest.approx(1.0, abs=1e-6)
    # Mass only on action 5 and 60
    assert target[5].item() == pytest.approx(0.5, abs=1e-6)
    assert target[60].item() == pytest.approx(0.5, abs=1e-6)
    assert target[150].item() == 0.0
    assert target[204].item() == 0.0


def test_target_uniform_over_legal_when_no_vp_legal():
    """When no VP-yielding action is legal, fall back to uniform over
    legal (no signal — better than NaN or all-zero)."""
    legal = torch.zeros(280, dtype=torch.bool)
    legal[150] = True     # BuildRoad
    legal[160] = True     # BuildRoad
    legal[204] = True     # EndTurn
    target = build_vp_prior_target(legal)
    assert target.sum().item() == pytest.approx(1.0, abs=1e-6)
    assert target[150].item() == pytest.approx(1/3, abs=1e-6)
    assert target[160].item() == pytest.approx(1/3, abs=1e-6)
    assert target[204].item() == pytest.approx(1/3, abs=1e-6)


def test_target_only_settlement_legal():
    """Only one VP action legal -> all mass on it."""
    legal = torch.zeros(280, dtype=torch.bool)
    legal[10] = True      # BuildSettlement (single VP-yielding)
    legal[150] = True     # BuildRoad
    legal[204] = True     # EndTurn
    target = build_vp_prior_target(legal)
    assert target[10].item() == pytest.approx(1.0, abs=1e-6)
    assert target[150].item() == 0.0
    assert target[204].item() == 0.0


def test_target_buydevcard_weighted_less_than_settle():
    """BuyDevCard (1/14 VP) gets less mass than BuildSettlement (1.0 VP)
    when both legal. The relative weight is the ratio of their
    CLASS_VP_VALUE entries."""
    legal = torch.zeros(280, dtype=torch.bool)
    legal[20] = True      # BuildSettlement
    legal[226] = True     # BuyDevCard
    target = build_vp_prior_target(legal)
    # Settlement weight = 1.0; BuyDevCard weight = CLASS_VP_VALUE[BUY_DEV_CARD]
    expected_dev_share = CLASS_VP_VALUE[ActionClass.BUY_DEV_CARD] / (
        1.0 + CLASS_VP_VALUE[ActionClass.BUY_DEV_CARD]
    )
    expected_settle_share = 1.0 - expected_dev_share
    assert target[20].item() == pytest.approx(expected_settle_share, abs=1e-6)
    assert target[226].item() == pytest.approx(expected_dev_share, abs=1e-6)


def test_target_works_on_batches():
    """build_vp_prior_target should handle a batch [B, 280] mask."""
    legal = torch.zeros(3, 280, dtype=torch.bool)
    legal[0, 5] = True; legal[0, 60] = True  # 2 VP actions
    legal[1, 150] = True; legal[1, 204] = True  # 0 VP, fallback uniform
    legal[2, 10] = True  # 1 VP
    target = build_vp_prior_target(legal)
    assert target.shape == (3, 280)
    assert target[0].sum().item() == pytest.approx(1.0, abs=1e-6)
    assert target[1].sum().item() == pytest.approx(1.0, abs=1e-6)
    assert target[2].sum().item() == pytest.approx(1.0, abs=1e-6)
    assert target[2, 10].item() == pytest.approx(1.0, abs=1e-6)
