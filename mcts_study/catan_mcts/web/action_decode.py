"""Decode raw engine action ids into UI-friendly objects.

Action space (v2, 280 actions):
  0..53    BuildSettlement(v)   target = vertex
  54..107  BuildCity(v)         target = vertex (v = a-54)
  108..179 BuildRoad(e)         target = edge   (e = a-108)
  180..198 MoveRobber(h)        target = hex    (h = a-180)
  199..203 Discard(res)         non-spatial
  204      EndTurn              non-spatial
  205      RollDice             non-spatial
  206..225 TradeBank            non-spatial
  226      BuyDevCard           non-spatial
  227      PlayKnight           non-spatial
  228      PlayRoadBuilding     non-spatial
  229..233 PlayMonopoly         non-spatial
  234..258 PlayYearOfPlenty     non-spatial
  259      PlayVpCard           non-spatial
  260..279 ProposeTrade         non-spatial
"""
from __future__ import annotations

from catan_mcts.web.serializers import action_desc


def _kind_and_target(a: int):
    if 0 <= a < 54:    return "build_settlement", a
    if 54 <= a < 108:  return "build_city", a - 54
    if 108 <= a < 180: return "build_road", a - 108
    if 180 <= a < 199: return "move_robber", a - 180
    if 199 <= a < 204: return "discard", None
    if a == 204:       return "end_turn", None
    if a == 205:       return "roll", None
    if 206 <= a < 226: return "trade_bank", None
    if a == 226:       return "buy_dev", None
    if 227 <= a <= 259: return "play_dev", None
    if 260 <= a < 280: return "propose_trade", None
    return "unknown", None


def decode(a: int) -> dict:
    a = int(a)
    kind, target = _kind_and_target(a)
    return {"id": a, "label": action_desc(a), "kind": kind, "target": target}


def decode_many(actions) -> list[dict]:
    return [decode(int(a)) for a in actions]
