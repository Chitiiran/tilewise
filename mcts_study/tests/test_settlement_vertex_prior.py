"""Tests for Cand 1 (settlement-vertex pip-weighted prior) of the
loss-augmentation roadmap.

Per chat 2026-05-12:
  - Pure pip score (no resource VP weighting) — vertex_score[v] = sum of
    pip[h] for h adjacent to v. Robber ignored (Option A).
  - Fires whenever ANY settlement action is legal (no phase gating).
  - lambda_settle = 0.20 (Cell 2 first run).

Engine action ID layout (cited actions.rs:121): BuildSettlement = action_id
in 0..53, where action_id == vertex_id directly.

hex_features layout (cited observation.rs:75-86):
  hex_features[h, 0..4] = wood/brick/sheep/wheat/ore one-hot
  hex_features[h, 5]    = (dice_num - 7) / 5  (normalized)
  hex_features[h, 6]    = robber flag
  hex_features[h, 7]    = desert flag

Pip values (cited standard Catan):
  2,12 -> 1 dot;  3,11 -> 2;  4,10 -> 3;  5,9 -> 4;  6,8 -> 5
  desert -> 0
"""
from __future__ import annotations

import numpy as np
import torch
import pytest

from catan_gnn.settlement_vertex_prior import (
    PIP_BY_DICE,
    hex_features_to_pip,
    compute_vertex_score,
    build_settlement_target,
    settlement_prior_loss,
)


# ============ PIP_BY_DICE table ============

def test_pip_table_standard_catan():
    assert PIP_BY_DICE[2] == 1
    assert PIP_BY_DICE[3] == 2
    assert PIP_BY_DICE[4] == 3
    assert PIP_BY_DICE[5] == 4
    assert PIP_BY_DICE[6] == 5
    assert PIP_BY_DICE[8] == 5
    assert PIP_BY_DICE[9] == 4
    assert PIP_BY_DICE[10] == 3
    assert PIP_BY_DICE[11] == 2
    assert PIP_BY_DICE[12] == 1
    assert PIP_BY_DICE[0] == 0  # desert


# ============ hex_features_to_pip ============

def test_hex_features_to_pip_normal_hex():
    """Hex with dice=8: hex_features[h, 5] = (8-7)/5 = 0.2, desert=0.
    Expected pip = 5."""
    hex_features = torch.zeros(1, 19, 8)
    hex_features[0, 5, 5] = (8 - 7) / 5  # dice_num=8 at hex 5
    pip = hex_features_to_pip(hex_features)
    assert pip.shape == (1, 19)
    assert pip[0, 5].item() == pytest.approx(5.0)


def test_hex_features_to_pip_desert():
    """Desert hex (feature[7]=1.0) yields 0 pip regardless of dice."""
    hex_features = torch.zeros(1, 19, 8)
    hex_features[0, 3, 7] = 1.0  # desert at hex 3
    hex_features[0, 3, 5] = (8 - 7) / 5  # desert with leftover dice value
    pip = hex_features_to_pip(hex_features)
    assert pip[0, 3].item() == pytest.approx(0.0)


def test_hex_features_to_pip_all_zero_when_no_dice():
    """If both desert flag = 0 and dice norm = 0 -> implies dice_num = 7.
    Per standard Catan, dice 7 never appears on a hex tile, so this is
    effectively desert. Should yield 0."""
    hex_features = torch.zeros(1, 19, 8)
    # All zeros — round((0 * 5) + 7) = 7 -> PIP_BY_DICE[7] should be 0
    pip = hex_features_to_pip(hex_features)
    # Every hex should be 0
    assert torch.all(pip == 0).item()


def test_hex_features_to_pip_batched():
    """Two-sample batch with different boards."""
    hex_features = torch.zeros(2, 19, 8)
    # Sample 0: hex 0 has dice=6 (pip=5)
    hex_features[0, 0, 5] = (6 - 7) / 5
    # Sample 1: hex 5 has dice=2 (pip=1)
    hex_features[1, 5, 5] = (2 - 7) / 5
    pip = hex_features_to_pip(hex_features)
    assert pip.shape == (2, 19)
    assert pip[0, 0].item() == pytest.approx(5.0)
    assert pip[0, 5].item() == pytest.approx(0.0)
    assert pip[1, 0].item() == pytest.approx(0.0)
    assert pip[1, 5].item() == pytest.approx(1.0)


# ============ compute_vertex_score ============

def test_vertex_score_single_hex_propagates_to_6_vertices():
    """A hex with pip=5 contributes 5 to each of its 6 adjacent vertices."""
    from catan_gnn.adjacency import HEX_TO_VERTICES
    hex_pip = torch.zeros(1, 19)
    hex_pip[0, 0] = 5.0  # hex 0
    vertex_score = compute_vertex_score(hex_pip)
    assert vertex_score.shape == (1, 54)
    # Each vertex adjacent to hex 0 should have score 5
    for v in HEX_TO_VERTICES[0]:
        assert vertex_score[0, v].item() == pytest.approx(5.0), (
            f"vertex {v} adjacent to hex 0 should have score 5"
        )


def test_vertex_score_sums_across_adjacent_hexes():
    """A vertex adjacent to multiple hexes sums their pips."""
    from catan_gnn.adjacency import HEX_TO_VERTICES
    # Find a vertex that's adjacent to at least 2 hexes
    v2h: dict[int, list[int]] = {}
    for h, vs in enumerate(HEX_TO_VERTICES):
        for v in vs:
            v2h.setdefault(int(v), []).append(h)
    multi_v = next(v for v, hs in v2h.items() if len(hs) >= 2)
    adjacent_hexes = v2h[multi_v]

    hex_pip = torch.zeros(1, 19)
    for i, h in enumerate(adjacent_hexes):
        hex_pip[0, h] = float(i + 1)  # distinct values

    expected = sum(i + 1 for i in range(len(adjacent_hexes)))
    vertex_score = compute_vertex_score(hex_pip)
    assert vertex_score[0, multi_v].item() == pytest.approx(expected)


def test_vertex_score_zero_when_all_hexes_zero():
    hex_pip = torch.zeros(2, 19)
    vertex_score = compute_vertex_score(hex_pip)
    assert vertex_score.shape == (2, 54)
    assert torch.all(vertex_score == 0).item()


# ============ build_settlement_target ============

def test_target_normalized_over_legal_settlements():
    """Target should sum to 1 over legal settlement vertices, zero elsewhere."""
    vertex_score = torch.zeros(1, 54)
    vertex_score[0, 5] = 10.0   # high pip
    vertex_score[0, 10] = 5.0   # medium pip
    vertex_score[0, 15] = 1.0   # low pip
    legal_settle = torch.zeros(1, 54, dtype=torch.bool)
    legal_settle[0, 5] = True
    legal_settle[0, 10] = True
    # Note: vertex 15 is NOT legal, but it has pip — should be ignored.

    target = build_settlement_target(vertex_score, legal_settle)
    assert target.shape == (1, 54)
    assert target[0, 5].item() == pytest.approx(10.0 / 15.0)
    assert target[0, 10].item() == pytest.approx(5.0 / 15.0)
    assert target[0, 15].item() == pytest.approx(0.0)
    assert target[0].sum().item() == pytest.approx(1.0)


def test_target_uniform_fallback_when_all_legal_zero_score():
    """If all legal vertices have score=0 (e.g., desert-only board), fall
    back to uniform over legal."""
    vertex_score = torch.zeros(1, 54)  # all zero
    legal_settle = torch.zeros(1, 54, dtype=torch.bool)
    legal_settle[0, 3] = True
    legal_settle[0, 7] = True
    legal_settle[0, 20] = True

    target = build_settlement_target(vertex_score, legal_settle)
    assert target[0, 3].item() == pytest.approx(1/3)
    assert target[0, 7].item() == pytest.approx(1/3)
    assert target[0, 20].item() == pytest.approx(1/3)
    assert target[0].sum().item() == pytest.approx(1.0)


def test_target_only_highest_pip_when_one_legal():
    """If only one legal settlement, target = one-hot on it."""
    vertex_score = torch.zeros(1, 54)
    vertex_score[0, 12] = 7.0
    legal_settle = torch.zeros(1, 54, dtype=torch.bool)
    legal_settle[0, 12] = True
    target = build_settlement_target(vertex_score, legal_settle)
    assert target[0, 12].item() == pytest.approx(1.0)


# ============ settlement_prior_loss ============

def test_loss_zero_when_no_legal_settlement():
    """If no settlement action is legal in any sample, loss is 0."""
    p_logits = torch.zeros(2, 280)
    legal_mask = torch.zeros(2, 280, dtype=torch.bool)
    legal_mask[0, 100] = True   # BuildRoad
    legal_mask[1, 204] = True   # EndTurn
    hex_features = torch.zeros(2, 19, 8)
    loss = settlement_prior_loss(p_logits, legal_mask, hex_features)
    assert loss.item() == pytest.approx(0.0)


def test_loss_minimized_when_logits_match_pip_target():
    """If the model's logits put all mass on the highest-pip legal
    settlement, the loss should be near zero."""
    p_logits = torch.full((1, 280), -10.0)
    p_logits[0, 5] = 10.0    # huge logit on vertex 5

    legal_mask = torch.zeros(1, 280, dtype=torch.bool)
    legal_mask[0, 5] = True
    legal_mask[0, 10] = True

    hex_features = torch.zeros(1, 19, 8)
    # Make vertex 5 the highest-pip via hex 0 (vertex 5 is NOT adjacent to hex 0
    # per HEX_TO_VERTICES[0]=[0,4,8,12,7,3], but let's pick a vertex
    # that IS adjacent. Vertex 8 is adjacent to hex 0.
    # Actually we need the legal one to be the high-pip one. Use vertex 8.
    p_logits = torch.full((1, 280), -10.0)
    p_logits[0, 8] = 10.0
    legal_mask = torch.zeros(1, 280, dtype=torch.bool)
    legal_mask[0, 8] = True
    legal_mask[0, 30] = True  # also legal but not adjacent to hex 0
    hex_features[0, 0, 5] = (6 - 7) / 5.0  # hex 0 has dice=6 (pip=5)
    # Result: vertex 8 has score 5; vertex 30 has score 0
    # target = [1.0 on v=8, 0.0 elsewhere among legal]
    # logits put huge mass on v=8 -> CE close to 0

    loss = settlement_prior_loss(p_logits, legal_mask, hex_features)
    # Should be small (model picks the right vertex)
    assert loss.item() < 0.01, f"expected near-zero loss, got {loss.item()}"


def test_loss_higher_when_logits_disagree_with_pip_target():
    """Conversely, when logits put mass on a low-pip vertex, loss is large."""
    p_logits = torch.full((1, 280), -10.0)
    p_logits[0, 30] = 10.0   # logit on the low-pip vertex
    legal_mask = torch.zeros(1, 280, dtype=torch.bool)
    legal_mask[0, 8] = True
    legal_mask[0, 30] = True
    hex_features = torch.zeros(1, 19, 8)
    hex_features[0, 0, 5] = (6 - 7) / 5.0  # hex 0 has dice=6 (pip=5)
    # Target = [1.0 on v=8, 0.0 on v=30 among legal]
    # Logits put mass on v=30 -> -log(near-0) is large
    loss = settlement_prior_loss(p_logits, legal_mask, hex_features)
    assert loss.item() > 5.0, f"expected large loss, got {loss.item()}"


def test_loss_batched_skips_no_legal_samples():
    """A batch with mixed has-legal-settle samples averages over the firing samples only."""
    B = 3
    p_logits = torch.zeros(B, 280)
    legal_mask = torch.zeros(B, 280, dtype=torch.bool)
    # Sample 0: settlement legal at v=10
    legal_mask[0, 10] = True
    # Sample 1: only EndTurn legal, no settlement
    legal_mask[1, 204] = True
    # Sample 2: settlement legal at v=20
    legal_mask[2, 20] = True

    hex_features = torch.zeros(B, 19, 8)
    loss = settlement_prior_loss(p_logits, legal_mask, hex_features)
    # Should be finite (two samples contribute, one is skipped)
    assert torch.isfinite(loss).item()
    assert loss.item() >= 0
