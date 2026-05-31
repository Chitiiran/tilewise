"""Discover available bot types + checkpoints, and build bot instances.

A "bot" here is anything with a `.step(state) -> int` method. We reuse the
existing bot classes; GNN types are loaded lazily so importing this module
never forces torch.
"""
from __future__ import annotations

import random
from pathlib import Path


class _RandomBot:
    """Picks a uniformly-random legal action."""
    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def step(self, state) -> int:
        legal = state.legal_actions()
        if not legal:
            raise RuntimeError("_RandomBot: no legal actions")
        return self._rng.choice(legal)


def list_types() -> list[dict]:
    """Bot types selectable in the lobby. `needs_checkpoint` drives the UI."""
    return [
        {"id": "Random", "label": "Random", "needs_checkpoint": False},
        {"id": "Greedy", "label": "Greedy baseline", "needs_checkpoint": False},
        {"id": "LookaheadMctsV3", "label": "Lookahead MCTS v3", "needs_checkpoint": False},
        {"id": "PureGnn", "label": "Pure GNN", "needs_checkpoint": True},
        {"id": "GnnMcts", "label": "GNN + MCTS", "needs_checkpoint": True},
    ]


def build(spec: dict, *, game, seed: int):
    """Construct a bot from a spec like {"type": "Random"} or
    {"type": "PureGnn", "checkpoint": "/abs/path.pt"}.

    `game` is a CatanGame (needed by MCTS/GNN bots); `seed` seeds the bot.
    """
    t = spec.get("type")
    if t == "Random":
        return _RandomBot(seed=seed)
    if t == "Greedy":
        from catan_mcts.bots import GreedyBaselineBot
        return GreedyBaselineBot(seed=seed)
    if t == "LookaheadMctsV3":
        from catan_mcts.players_v3 import build_lookahead_mcts_v3
        return build_lookahead_mcts_v3(game, seed=seed)
    if t in ("PureGnn", "GnnMcts"):
        return _build_gnn_bot(spec, game=game, seed=seed)
    raise ValueError(f"unknown bot type: {t!r}")


def _build_gnn_bot(spec, *, game, seed):  # implemented in Task 5
    raise NotImplementedError("GNN bot construction lands in Task 5")
