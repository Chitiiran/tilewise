"""Engine-faithful prediction of who would accept a ProposeTrade.

Mirrors catan_engine/src/rules.rs:347-374: scan opponents in seat order
(current+1,+2,+3); the first holding >=1 of `get` accepts a 1-for-1 swap.
"""
from __future__ import annotations

PROPOSE_TRADE_BASE = 260


def decode_propose_trade(action: int) -> tuple[int, int]:
    """action 260..279 -> (give_resource_idx, get_resource_idx)."""
    idx = int(action) - PROPOSE_TRADE_BASE
    if not (0 <= idx < 20):
        raise ValueError(f"not a ProposeTrade action: {action}")
    give = idx // 4
    get_in_others = idx % 4
    others = [r for r in range(5) if r != give]
    return give, others[get_in_others]


def first_acceptor(current_player: int, give: int, get: int, hands) -> int:
    """Seat of the first opponent (current+1,+2,+3) holding >=1 of `get`, else -1."""
    for offset in range(1, 4):
        opp = (current_player + offset) % 4
        if hands[opp][get] >= 1:
            return opp
    return -1


def would_match_human(current_player: int, action: int, hands, human_seat: int) -> bool:
    """True iff the engine would auto-match the human for this ProposeTrade."""
    give, get = decode_propose_trade(action)
    return first_acceptor(current_player, give, get, hands) == human_seat
