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


def list_checkpoints(checkpoints_dir) -> list[dict]:
    """Recursively list *.pt files under `checkpoints_dir` (sorted by name)."""
    root = Path(checkpoints_dir)
    if not root.exists():
        return []
    out = []
    for p in sorted(root.rglob("*.pt")):
        out.append({"name": p.name, "path": str(p)})
    return out


def _load_gnn_model(checkpoint: str, *, hidden_dim: int, num_layers: int, device: str):
    """Load a GnnModel from a .pt checkpoint (handles {'model_state': ...} wrappers)."""
    if not Path(checkpoint).exists():
        raise ValueError(f"checkpoint not found: {checkpoint}")
    import torch
    from catan_gnn.gnn_model import GnnModel
    try:
        obj = torch.load(checkpoint, map_location=device, weights_only=False)
        state = obj["model_state"] if isinstance(obj, dict) and "model_state" in obj else obj
        model = GnnModel(hidden_dim=hidden_dim, num_layers=num_layers)
        model.load_state_dict(state)
    except Exception as e:
        raise ValueError(f"failed to load checkpoint {checkpoint!r}: {e}") from e
    return model.to(device).eval()


def _build_gnn_bot(spec, *, game, seed):
    checkpoint = spec.get("checkpoint")
    if not checkpoint:
        raise ValueError("GNN bot requires a 'checkpoint' path")
    device = spec.get("device", "cpu")
    hidden_dim = int(spec.get("hidden_dim", 32))
    num_layers = int(spec.get("num_layers", 2))
    model = _load_gnn_model(checkpoint, hidden_dim=hidden_dim,
                            num_layers=num_layers, device=device)
    if spec["type"] == "PureGnn":
        from catan_mcts.bots_gnn import PureGnnBot
        return PureGnnBot(model=model, device=device)
    from catan_mcts.experiments.e10e_gnn_mcts import build_gnn_mcts_bot
    sims = int(spec.get("sims", 200))
    return build_gnn_mcts_bot(game, model, sims=sims, seed=seed, device=device)
