"""Unit tests for Cand 7: action-class-balanced policy target.

The transformation: for each legal action a in a sample, divide its
target weight by the number of legal actions in the same action_class,
then renormalize so the per-sample target sums to 1.

This makes a single road action's gradient comparable to a single
trade action's gradient regardless of how many of each class are legal.

Cited spec: docs/superpowers/specs/2026-05-09-loss-augmentation-design.md
section "Candidate 7" and plan ordering doc Cell 2.
"""
from __future__ import annotations

import torch
import pytest

from catan_gnn.action_classes import ACTION_CLASS_ID, NUM_ACTION_CLASSES


@pytest.fixture(scope="module")
def action_class_id():
    """The 280-long int tensor mapping each action_id to its class index."""
    return ACTION_CLASS_ID


# ---------------------------------------------------------------------
# ACTION_CLASS_ID basic properties
# ---------------------------------------------------------------------

def test_action_class_id_shape(action_class_id):
    assert action_class_id.shape == (280,)
    assert action_class_id.dtype == torch.long


def test_action_class_id_range(action_class_id):
    """Each entry must be a valid class index."""
    assert (action_class_id >= 0).all()
    assert (action_class_id < NUM_ACTION_CLASSES).all()


def test_action_class_id_boundaries(action_class_id):
    """Spot-check a few action-id → class mappings from action_classes.py
    docstring's cited ranges."""
    from catan_gnn.action_classes import ActionClass, ACTION_CLASS
    # BuildSettlement (0..53)
    assert ACTION_CLASS[0] == ActionClass.BUILD_SETTLEMENT
    assert ACTION_CLASS[53] == ActionClass.BUILD_SETTLEMENT
    # BuildCity (54..107)
    assert ACTION_CLASS[54] == ActionClass.BUILD_CITY
    # BuildRoad (108..179)
    assert ACTION_CLASS[108] == ActionClass.BUILD_ROAD
    assert ACTION_CLASS[179] == ActionClass.BUILD_ROAD
    # ProposeTrade (260..279)
    assert ACTION_CLASS[260] == ActionClass.PROPOSE_TRADE
    assert ACTION_CLASS[279] == ActionClass.PROPOSE_TRADE
    # ACTION_CLASS_ID and ACTION_CLASS must agree
    for a in [0, 53, 54, 108, 179, 204, 260, 279]:
        expected = list(ActionClass).index(ACTION_CLASS[a])
        assert action_class_id[a].item() == expected, (
            f"action {a}: ACTION_CLASS_ID[{a}]={action_class_id[a].item()} "
            f"!= index of {ACTION_CLASS[a]} ({expected})"
        )


# ---------------------------------------------------------------------
# class_balanced_target transformation
# ---------------------------------------------------------------------

def test_class_balanced_target_5roads_1city_uniform_visits():
    """5 legal roads (action_ids 108..112), 1 legal city (action_id 54).
    MCTS visits split 50/50 (0.5 mass on roads collectively, 0.5 on city).
    Specifically: each road gets 0.1, city gets 0.5.

    After class-balancing:
      - Each road's adjusted weight: 0.1 / 5  = 0.02; total road = 0.10
      - City's adjusted weight:      0.5 / 1  = 0.50; total city = 0.50
      - Renormalized: total mass becomes 0.10 + 0.50 = 0.60
      - Final per-road: 0.02/0.60 = 0.0333..; per-city: 0.50/0.60 = 0.8333..

    Notable: this DOES NOT equalize class mass — it just divides each
    sample's contribution by class-count, then renormalizes. The CITY
    side already had a 1/1 class denominator so its mass is preserved
    while roads get spread out.
    """
    from catan_gnn.train import class_balanced_target

    legal = torch.zeros(1, 280, dtype=torch.bool)
    legal[0, 108:113] = True  # 5 roads
    legal[0, 54] = True       # 1 city

    target = torch.zeros(1, 280)
    target[0, 108:113] = 0.1  # 5 roads, 0.1 each = 0.5 total
    target[0, 54] = 0.5

    out = class_balanced_target(target, legal)

    # Shape preserved
    assert out.shape == target.shape
    # Sums to 1
    assert torch.allclose(out.sum(dim=-1), torch.tensor([1.0]), atol=1e-5)
    # Each road has the same adjusted weight (symmetric)
    road_vals = out[0, 108:113]
    assert torch.allclose(road_vals, road_vals[0].expand_as(road_vals), atol=1e-6)
    # City weight ratio to single road weight = 25× (0.5/0.02 = 25 before
    # normalize, ratio preserved after)
    expected_ratio = (0.5 / 1) / (0.1 / 5)  # = 25
    assert abs(out[0, 54].item() / out[0, 108].item() - expected_ratio) < 1e-4


def test_class_balanced_target_idempotent_one_per_class():
    """When every class has exactly 1 legal action, class_count = 1
    everywhere → division is a no-op → output == input (after renorm,
    which is identity since target already sums to 1)."""
    from catan_gnn.train import class_balanced_target

    legal = torch.zeros(1, 280, dtype=torch.bool)
    # One action per class (pick the first in each block)
    legal[0, [0, 54, 108, 180, 199, 204, 205, 206, 226, 227, 228, 229, 234, 259, 260]] = True
    target = torch.zeros(1, 280)
    # Split uniformly
    target[0, legal[0]] = 1.0 / int(legal[0].sum())

    out = class_balanced_target(target, legal)
    assert torch.allclose(out, target, atol=1e-6)


def test_class_balanced_target_all_zero_target_stays_zero():
    """If target is all zeros (degenerate sample), output stays zero
    rather than NaN. The masked policy loss handles zero targets fine."""
    from catan_gnn.train import class_balanced_target

    legal = torch.zeros(1, 280, dtype=torch.bool)
    legal[0, 108:113] = True
    target = torch.zeros(1, 280)

    out = class_balanced_target(target, legal)
    assert torch.isfinite(out).all()
    assert out.sum().item() == 0.0


def test_class_balanced_target_illegal_actions_stay_zero():
    """Illegal actions must have output weight = 0 even if their input
    weight was 0 (which it always is per dataset invariant)."""
    from catan_gnn.train import class_balanced_target

    legal = torch.zeros(2, 280, dtype=torch.bool)
    legal[0, 108:113] = True
    legal[0, 54] = True
    legal[1, 0:5] = True

    target = torch.zeros(2, 280)
    target[0, 108:113] = 0.1
    target[0, 54] = 0.5
    target[1, 0:5] = 0.2

    out = class_balanced_target(target, legal)
    # All illegal positions must be exactly 0
    assert (out[~legal] == 0).all()


def test_class_balanced_target_batched():
    """Batched input: per-sample renormalization is independent."""
    from catan_gnn.train import class_balanced_target

    legal = torch.zeros(3, 280, dtype=torch.bool)
    target = torch.zeros(3, 280)

    # Sample 0: 5 roads only
    legal[0, 108:113] = True
    target[0, 108:113] = 0.2

    # Sample 1: 1 city only
    legal[1, 54] = True
    target[1, 54] = 1.0

    # Sample 2: 3 settles + 2 cities (mixed)
    legal[2, 0:3] = True
    legal[2, 54:56] = True
    target[2, 0:3] = 0.2
    target[2, 54:56] = 0.2

    out = class_balanced_target(target, legal)
    # All samples sum to 1 (or 0 for degenerate, but none here are)
    sums = out.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(3), atol=1e-5)

    # Sample 1 unchanged (single legal action → trivially renormalizes)
    assert out[1, 54].item() == pytest.approx(1.0, abs=1e-6)


def test_class_balanced_target_reduces_road_gradient_share():
    """The motivating case: roads dominate the action-id count, so they
    inflate gradient share. After class-balancing, 5 legal roads + 1
    legal trade should give each trade more per-action weight than each
    road (which is the whole point of Cand 7).
    """
    from catan_gnn.train import class_balanced_target

    legal = torch.zeros(1, 280, dtype=torch.bool)
    legal[0, 108:113] = True   # 5 roads
    legal[0, 260] = True       # 1 trade

    # Uniform MCTS visits: each legal action gets equal mass.
    target = torch.zeros(1, 280)
    target[0, legal[0]] = 1.0 / 6

    out = class_balanced_target(target, legal)
    # Before Cand 7: every legal action = 1/6 ≈ 0.167
    # After Cand 7: each road gets (1/6)/5 = 1/30, trade gets (1/6)/1 = 1/6
    #   Renormalize: total = 5*(1/30) + 1*(1/6) = 1/6 + 1/6 = 2/6
    #   Per road: (1/30) / (2/6) = 6/60 = 0.1
    #   Per trade: (1/6) / (2/6) = 0.5
    assert out[0, 108].item() == pytest.approx(0.1, abs=1e-4)
    assert out[0, 260].item() == pytest.approx(0.5, abs=1e-4)
    # Trade weight > 4× any single road weight
    assert out[0, 260] > 4 * out[0, 108]
