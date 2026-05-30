"""Action-class taxonomy + VP-yield weights for Cand 8 of the
loss-augmentation roadmap.

Each of the 280 engine actions belongs to exactly one ActionClass.
Each class has a CLASS_VP_VALUE in [0, 1] representing the expected
direct VP yield of taking that action in v3 (vp_target=5, bonuses=False).

Cited rules.rs for VP-grant points:
  - rules.rs:97   Setup1Place settlement -> +1 VP
  - rules.rs:121  Setup2Place settlement -> +1 VP
  - rules.rs:181  Main-phase BuildSettlement -> +1 VP
  - rules.rs:210  BuildCity -> +1 VP (settlement was 1VP, city is 2VP, net +1)
  - rules.rs:266  PlayVpCard -> +1 VP

Cited state.rs:230 DEV_CARD_DECK_STANDARD = [14, 2, 2, 2, 5] meaning
14 Knight + 2 RoadBuilding + 2 Monopoly + 2 YearOfPlenty + 5 VP = 25
total. BuyDevCard's expected VP = 5/25 = 0.20.

Action ID ranges (cited actions.rs:121-127, 49-58):
  BuildSettlement:    0..53
  BuildCity:         54..107
  BuildRoad:        108..179
  MoveRobber:       180..198
  Discard:          199..203
  EndTurn:              204
  RollDice:             205
  TradeBank:        206..225
  BuyDevCard:           226
  PlayKnight:           227
  PlayRoadBuilding:     228
  PlayMonopoly:     229..233
  PlayYearOfPlenty: 234..258
  PlayVpCard:           259
  ProposeTrade:     260..279
"""
from __future__ import annotations

from enum import Enum

import torch


class ActionClass(Enum):
    BUILD_SETTLEMENT = "BuildSettlement"
    BUILD_CITY = "BuildCity"
    BUILD_ROAD = "BuildRoad"
    MOVE_ROBBER = "MoveRobber"
    DISCARD = "Discard"
    END_TURN = "EndTurn"
    ROLL_DICE = "RollDice"
    TRADE_BANK = "TradeBank"
    BUY_DEV_CARD = "BuyDevCard"
    PLAY_KNIGHT = "PlayKnight"
    PLAY_ROAD_BUILDING = "PlayRoadBuilding"
    PLAY_MONOPOLY = "PlayMonopoly"
    PLAY_YEAR_OF_PLENTY = "PlayYearOfPlenty"
    PLAY_VP_CARD = "PlayVpCard"
    PROPOSE_TRADE = "ProposeTrade"


# Cited DEV_CARD_DECK_STANDARD: 14+2+2+2+5 = 25 cards, 5 are VP cards.
# Expected VP from drawing one card = 5/25 = 0.20.
_DEV_CARD_VP_PROB = 5.0 / 25.0


# Per ActionClass, the expected direct VP yield in v3 of taking that
# action (assuming bonuses=False so longest-road / largest-army don't
# award +2 VP).
CLASS_VP_VALUE: dict[ActionClass, float] = {
    ActionClass.BUILD_SETTLEMENT: 1.0,
    ActionClass.BUILD_CITY: 1.0,
    ActionClass.PLAY_VP_CARD: 1.0,
    ActionClass.BUY_DEV_CARD: _DEV_CARD_VP_PROB,
    # All others — no direct VP in v3.
    ActionClass.BUILD_ROAD: 0.0,
    ActionClass.MOVE_ROBBER: 0.0,
    ActionClass.DISCARD: 0.0,
    ActionClass.END_TURN: 0.0,
    ActionClass.ROLL_DICE: 0.0,
    ActionClass.TRADE_BANK: 0.0,
    ActionClass.PLAY_KNIGHT: 0.0,
    ActionClass.PLAY_ROAD_BUILDING: 0.0,
    ActionClass.PLAY_MONOPOLY: 0.0,
    ActionClass.PLAY_YEAR_OF_PLENTY: 0.0,
    ActionClass.PROPOSE_TRADE: 0.0,
}


def _build_action_class_table() -> list[ActionClass]:
    """Construct the 280-entry table at module load. Each action_id
    maps to its ActionClass."""
    tbl: list[ActionClass | None] = [None] * 280
    for a in range(0, 54):
        tbl[a] = ActionClass.BUILD_SETTLEMENT
    for a in range(54, 108):
        tbl[a] = ActionClass.BUILD_CITY
    for a in range(108, 180):
        tbl[a] = ActionClass.BUILD_ROAD
    for a in range(180, 199):
        tbl[a] = ActionClass.MOVE_ROBBER
    for a in range(199, 204):
        tbl[a] = ActionClass.DISCARD
    tbl[204] = ActionClass.END_TURN
    tbl[205] = ActionClass.ROLL_DICE
    for a in range(206, 226):
        tbl[a] = ActionClass.TRADE_BANK
    tbl[226] = ActionClass.BUY_DEV_CARD
    tbl[227] = ActionClass.PLAY_KNIGHT
    tbl[228] = ActionClass.PLAY_ROAD_BUILDING
    for a in range(229, 234):
        tbl[a] = ActionClass.PLAY_MONOPOLY
    for a in range(234, 259):
        tbl[a] = ActionClass.PLAY_YEAR_OF_PLENTY
    tbl[259] = ActionClass.PLAY_VP_CARD
    for a in range(260, 280):
        tbl[a] = ActionClass.PROPOSE_TRADE
    assert all(x is not None for x in tbl), "ACTION_CLASS table has holes"
    return tbl  # type: ignore[return-value]


ACTION_CLASS: list[ActionClass] = _build_action_class_table()


# Cand 7: integer class id per action, for fast batched scatter ops.
# Each action_id (0..279) -> class index in [0, NUM_ACTION_CLASSES).
NUM_ACTION_CLASSES = len(ActionClass)
_CLASS_TO_IDX = {c: i for i, c in enumerate(ActionClass)}
ACTION_CLASS_ID: torch.Tensor = torch.tensor(
    [_CLASS_TO_IDX[ACTION_CLASS[a]] for a in range(280)],
    dtype=torch.long,
)


# Precomputed VP-value vector indexed by action_id. Shape (280,) on CPU.
# build_vp_prior_target uses this to mask & weight in one shot.
_ACTION_VP_VALUE_TENSOR = torch.tensor(
    [CLASS_VP_VALUE[ACTION_CLASS[a]] for a in range(280)],
    dtype=torch.float32,
)


def build_vp_prior_target(legal_mask: torch.Tensor) -> torch.Tensor:
    """Build a per-sample (or per-batch) VP-prior target distribution.

    For each sample:
      - Take CLASS_VP_VALUE for each action_id.
      - Mask out illegal actions (set their weight to 0).
      - If any VP-yielding action is legal, normalize to a distribution
        over those VP actions (weighted by CLASS_VP_VALUE).
      - If NO VP-yielding action is legal, fall back to uniform over
        all legal actions (signal-less; better than NaN or all-zero).

    Args:
        legal_mask: bool tensor of shape (..., 280). Last dim is action.

    Returns:
        Target distribution, same shape as legal_mask, dtype float32.
        Each sample sums to 1.0.
    """
    if legal_mask.dtype != torch.bool:
        legal_mask = legal_mask.bool()

    vp = _ACTION_VP_VALUE_TENSOR.to(legal_mask.device)
    # Broadcast vp across batch.
    weighted = legal_mask.to(torch.float32) * vp  # (..., 280)
    vp_sum = weighted.sum(dim=-1, keepdim=True)   # (..., 1)
    has_vp = vp_sum > 0

    # Where vp_sum > 0: normalize weighted by vp_sum.
    # Where vp_sum == 0: fall back to uniform over legal.
    legal_count = legal_mask.to(torch.float32).sum(dim=-1, keepdim=True).clamp(min=1)
    uniform = legal_mask.to(torch.float32) / legal_count

    # Use broadcasting to pick the right per-sample target.
    target = torch.where(has_vp, weighted / vp_sum.clamp(min=1e-12), uniform)
    return target
