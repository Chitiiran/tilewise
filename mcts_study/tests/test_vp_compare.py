"""Tests for Cand 10 (1-step VP comparison target swap) of the
loss-augmentation roadmap.

Per discussion 2026-05-11 in chat: in v3 (bonuses=False), an action's
1-step VP delta is FULLY determined by its action_class via
CLASS_VP_VALUE (cited rules.rs: all longest-road / largest-army VP
mutations are gated by `state.bonuses_enabled`). So the engine is not
needed to compute vp(a_model) vs vp(a_teacher) -- the comparison
reduces to CLASS_VP_VALUE[a_model] > CLASS_VP_VALUE[a_teacher].

The rule:
  For each training sample with teacher action a_teacher and model
  argmax action a_model:
    if CLASS_VP_VALUE[a_model] > CLASS_VP_VALUE[a_teacher]:
        target = one_hot(a_model)       # swap: reinforce model's choice
    else:
        target = policy_t (visit counts)  # keep teacher

vp_compare_swap_target operates per-batch on the masked logits, the
teacher target (visit-count distribution), and the legal mask. Returns
the new target distribution.

Tested invariants:
  - When no swap condition fires, target is identical to input policy_t.
  - When model picks BuildCity and teacher picks ProposeTrade, target
    becomes one-hot on the BuildCity action.
  - When model picks ProposeTrade and teacher picks BuildCity, target
    stays as policy_t (no swap because model is WORSE).
  - When both pick same-class actions, no swap.
"""
from __future__ import annotations

import numpy as np
import torch
import pytest

from catan_gnn.action_classes import ActionClass, CLASS_VP_VALUE
from catan_gnn.vp_compare import vp_compare_swap_target


ACTION_SPACE = 280

# Sentinel action IDs from the cited engine layout.
SETTLE_ID = 5         # BuildSettlement: VP=1.0
CITY_ID = 60          # BuildCity: VP=1.0
ROAD_ID = 150         # BuildRoad: VP=0.0
TRADE_BANK_ID = 210   # TradeBank: VP=0.0
DEV_BUY_ID = 226      # BuyDevCard: VP=0.20
PROPOSE_TRADE_ID = 270  # ProposeTrade: VP=0.0
END_TURN_ID = 204     # EndTurn: VP=0.0
PLAY_VP_ID = 259      # PlayVpCard: VP=1.0


def _one_hot(idx: int, n: int = ACTION_SPACE) -> torch.Tensor:
    t = torch.zeros(n, dtype=torch.float32)
    t[idx] = 1.0
    return t


def _make_legal(action_ids: list[int]) -> torch.Tensor:
    mask = torch.zeros(ACTION_SPACE, dtype=torch.bool)
    for a in action_ids:
        mask[a] = True
    return mask


def _make_logits_argmaxing_to(a: int) -> torch.Tensor:
    logits = torch.zeros(ACTION_SPACE, dtype=torch.float32)
    logits[a] = 10.0  # large positive, will be argmax
    return logits


# ============ Single-sample swap behavior ============

def test_swap_fires_when_model_picks_city_teacher_picks_trade():
    """Model: BuildCity (VP=1.0); teacher: ProposeTrade (VP=0.0).
    Rule: swap target to one-hot(City)."""
    logits = _make_logits_argmaxing_to(CITY_ID).unsqueeze(0)
    legal = _make_legal([CITY_ID, ROAD_ID, PROPOSE_TRADE_ID, END_TURN_ID]).unsqueeze(0)
    policy_t = _one_hot(PROPOSE_TRADE_ID).unsqueeze(0)  # teacher chose trade

    new_target, swap_count = vp_compare_swap_target(logits, policy_t, legal)
    assert new_target.shape == (1, ACTION_SPACE)
    # New target = one-hot on CITY_ID
    assert new_target[0, CITY_ID].item() == pytest.approx(1.0)
    assert new_target[0, PROPOSE_TRADE_ID].item() == pytest.approx(0.0)
    assert swap_count == 1


def test_no_swap_when_teacher_already_picks_vp_action():
    """Model: ProposeTrade (VP=0); teacher: BuildCity (VP=1.0).
    Rule: do not swap; keep teacher target."""
    logits = _make_logits_argmaxing_to(PROPOSE_TRADE_ID).unsqueeze(0)
    legal = _make_legal([CITY_ID, PROPOSE_TRADE_ID, END_TURN_ID]).unsqueeze(0)
    policy_t = _one_hot(CITY_ID).unsqueeze(0)

    new_target, swap_count = vp_compare_swap_target(logits, policy_t, legal)
    # Target unchanged
    assert torch.allclose(new_target, policy_t, atol=1e-6)
    assert swap_count == 0


def test_no_swap_when_same_class():
    """Both pick a BuildSettlement (different IDs, same class & VP=1.0).
    Rule: tie -> no swap."""
    logits = _make_logits_argmaxing_to(SETTLE_ID).unsqueeze(0)
    legal = _make_legal([SETTLE_ID, SETTLE_ID + 1, END_TURN_ID]).unsqueeze(0)
    policy_t = _one_hot(SETTLE_ID + 1).unsqueeze(0)  # teacher: different settle

    new_target, swap_count = vp_compare_swap_target(logits, policy_t, legal)
    assert torch.allclose(new_target, policy_t, atol=1e-6)
    assert swap_count == 0


def test_no_swap_when_both_non_vp():
    """Model: BuildRoad; teacher: TradeBank. Both VP=0.0.
    Rule: tie -> no swap."""
    logits = _make_logits_argmaxing_to(ROAD_ID).unsqueeze(0)
    legal = _make_legal([ROAD_ID, TRADE_BANK_ID, END_TURN_ID]).unsqueeze(0)
    policy_t = _one_hot(TRADE_BANK_ID).unsqueeze(0)

    new_target, swap_count = vp_compare_swap_target(logits, policy_t, legal)
    assert torch.allclose(new_target, policy_t, atol=1e-6)
    assert swap_count == 0


def test_swap_dev_card_vs_road():
    """Model: BuyDevCard (VP=0.20); teacher: BuildRoad (VP=0.0).
    Rule: swap (0.20 > 0.0)."""
    logits = _make_logits_argmaxing_to(DEV_BUY_ID).unsqueeze(0)
    legal = _make_legal([DEV_BUY_ID, ROAD_ID, END_TURN_ID]).unsqueeze(0)
    policy_t = _one_hot(ROAD_ID).unsqueeze(0)

    new_target, swap_count = vp_compare_swap_target(logits, policy_t, legal)
    assert new_target[0, DEV_BUY_ID].item() == pytest.approx(1.0)
    assert swap_count == 1


def test_no_swap_dev_card_vs_settlement():
    """Model: BuyDevCard (VP=0.20); teacher: BuildSettlement (VP=1.0).
    Rule: model worse, no swap."""
    logits = _make_logits_argmaxing_to(DEV_BUY_ID).unsqueeze(0)
    legal = _make_legal([DEV_BUY_ID, SETTLE_ID, END_TURN_ID]).unsqueeze(0)
    policy_t = _one_hot(SETTLE_ID).unsqueeze(0)

    new_target, swap_count = vp_compare_swap_target(logits, policy_t, legal)
    assert torch.allclose(new_target, policy_t, atol=1e-6)
    assert swap_count == 0


def test_teacher_action_inferred_from_argmax_of_policy_t():
    """Teacher is the argmax of policy_t (could be a peaked but not
    one-hot distribution, like a visit count vector)."""
    logits = _make_logits_argmaxing_to(CITY_ID).unsqueeze(0)
    legal = _make_legal([CITY_ID, PROPOSE_TRADE_ID, END_TURN_ID]).unsqueeze(0)
    # Visit count style: peaked on ProposeTrade but not zero on others
    policy_t = torch.zeros(1, ACTION_SPACE)
    policy_t[0, PROPOSE_TRADE_ID] = 0.7
    policy_t[0, END_TURN_ID] = 0.2
    policy_t[0, CITY_ID] = 0.1

    new_target, swap_count = vp_compare_swap_target(logits, policy_t, legal)
    # teacher_argmax = PROPOSE_TRADE_ID (0.7 > 0.2 > 0.1)
    # model_argmax = CITY_ID
    # CLASS_VP_VALUE[CITY] = 1.0 > CLASS_VP_VALUE[PROPOSE_TRADE] = 0.0
    # swap should fire
    assert new_target[0, CITY_ID].item() == pytest.approx(1.0)
    assert swap_count == 1


def test_model_argmax_must_be_legal():
    """If a_model's argmax over masked logits points at a legal action
    (it must — logits are masked to -inf for illegal), the swap
    behavior is well-defined."""
    logits = torch.zeros(1, ACTION_SPACE, dtype=torch.float32)
    logits[0, CITY_ID] = 5.0           # legal, will win argmax-over-legal
    logits[0, SETTLE_ID + 1] = 100.0   # ILLEGAL but raw-argmax winner
    legal = _make_legal([CITY_ID, PROPOSE_TRADE_ID]).unsqueeze(0)
    policy_t = _one_hot(PROPOSE_TRADE_ID).unsqueeze(0)

    new_target, swap_count = vp_compare_swap_target(logits, policy_t, legal)
    # Must mask before argmax: a_model = CITY_ID
    # Swap fires (City > Trade)
    assert new_target[0, CITY_ID].item() == pytest.approx(1.0)
    assert swap_count == 1


# ============ Batch behavior ============

def test_batch_mixed_swap_and_no_swap():
    """Batch of 3: only middle sample should swap."""
    B = 3
    logits = torch.zeros(B, ACTION_SPACE, dtype=torch.float32)
    legal = torch.zeros(B, ACTION_SPACE, dtype=torch.bool)
    policy_t = torch.zeros(B, ACTION_SPACE, dtype=torch.float32)

    # Sample 0: model=Road, teacher=Road -> no swap
    logits[0, ROAD_ID] = 10.0
    legal[0, [ROAD_ID, END_TURN_ID]] = True
    policy_t[0, ROAD_ID] = 1.0

    # Sample 1: model=City, teacher=Trade -> SWAP
    logits[1, CITY_ID] = 10.0
    legal[1, [CITY_ID, PROPOSE_TRADE_ID]] = True
    policy_t[1, PROPOSE_TRADE_ID] = 1.0

    # Sample 2: model=Trade, teacher=Settle -> no swap
    logits[2, PROPOSE_TRADE_ID] = 10.0
    legal[2, [SETTLE_ID, PROPOSE_TRADE_ID]] = True
    policy_t[2, SETTLE_ID] = 1.0

    new_target, swap_count = vp_compare_swap_target(logits, policy_t, legal)
    assert new_target.shape == (B, ACTION_SPACE)
    # Sample 0 unchanged
    assert torch.allclose(new_target[0], policy_t[0], atol=1e-6)
    # Sample 1 swapped to one-hot CITY
    assert new_target[1, CITY_ID].item() == pytest.approx(1.0)
    assert new_target[1, PROPOSE_TRADE_ID].item() == pytest.approx(0.0)
    # Sample 2 unchanged
    assert torch.allclose(new_target[2], policy_t[2], atol=1e-6)
    # Total swap count
    assert swap_count == 1


def test_each_target_still_sums_to_one():
    """Per-sample target sum invariant — preserved by the swap (one-hot
    sums to 1; visit-count distributions are pre-normalized)."""
    B = 3
    logits = torch.zeros(B, ACTION_SPACE, dtype=torch.float32)
    legal = torch.ones(B, ACTION_SPACE, dtype=torch.bool)  # all legal
    policy_t = torch.zeros(B, ACTION_SPACE, dtype=torch.float32)
    # Sample 0: visit-count peak on PROPOSE_TRADE
    policy_t[0, PROPOSE_TRADE_ID] = 0.6; policy_t[0, ROAD_ID] = 0.4
    logits[0, CITY_ID] = 10.0  # model picks City
    # Sample 1: visit-count peak on SETTLE
    policy_t[1, SETTLE_ID] = 1.0
    logits[1, ROAD_ID] = 10.0  # model picks Road
    # Sample 2: visit-count peak on CITY (teacher = model)
    policy_t[2, CITY_ID] = 1.0
    logits[2, CITY_ID] = 10.0
    new_target, _ = vp_compare_swap_target(logits, policy_t, legal)
    for i in range(B):
        assert new_target[i].sum().item() == pytest.approx(1.0, abs=1e-5), (
            f"sample {i} target sums to {new_target[i].sum()} not 1.0"
        )
