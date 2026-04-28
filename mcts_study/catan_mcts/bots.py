"""Baseline bots and rollout policy for MCTS experiments.

Bots interact only with OpenSpiel state types. They never touch `_engine` directly —
that abstraction is what `adapter.py` provides.

Action ID ranges (post-Phase-0):
  BuildSettlement: 0..54   (1 VP)
  BuildCity:       54..108 (2 VP, net +1 over upgraded settlement)
  BuildRoad:       108..180
  MoveRobber:      180..199
  Discard:         199..204
  EndTurn:         204
  RollDice:        205
"""
from __future__ import annotations

import random
from typing import Optional


def _action_priority(action_id: int) -> int:
    """Higher = bot prefers it. Crude VP-greedy ordering."""
    if 54 <= action_id < 108:
        return 100  # city — best
    if 0 <= action_id < 54:
        return 80   # settlement
    if 108 <= action_id < 180:
        return 50   # road (enables future settlements)
    if action_id == 205:  # roll
        return 10
    if action_id == 204:  # end turn
        return 1
    if 180 <= action_id < 199:
        return 5    # robber move (no preference among hexes here)
    if 199 <= action_id < 204:
        return 5    # discard
    return 0


class GreedyBaselineBot:
    """Picks the legal action with highest VP-greedy priority. Ties broken randomly."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def step(self, state) -> int:
        legal = state.legal_actions()
        if not legal:
            raise RuntimeError("GreedyBaselineBot: no legal actions in non-terminal state")
        best_priority = max(_action_priority(a) for a in legal)
        candidates = [a for a in legal if _action_priority(a) == best_priority]
        return self._rng.choice(candidates)


def heuristic_rollout(state, rng: Optional[random.Random] = None) -> int:
    """A single-action callable matching MCTS rollout-policy signature.

    Returns one action; the MCTS rollout loop calls this until terminal.
    """
    if rng is None:
        rng = random
    legal = state.legal_actions()
    if not legal:
        raise RuntimeError("heuristic_rollout: no legal actions")
    best_priority = max(_action_priority(a) for a in legal)
    candidates = [a for a in legal if _action_priority(a) == best_priority]
    return rng.choice(candidates) if isinstance(rng, random.Random) else random.choice(candidates)
