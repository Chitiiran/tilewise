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


def _infer_arch(state) -> tuple[int, int]:
    """Infer (hidden_dim, num_layers) from a GnnModel state_dict.

    The training library holds checkpoints of varying architectures (h16/h32/
    h64/h128, 2-4 layers), so we can't assume the GnnModel defaults. We read:
      - num_layers from the highest `body.convs.<N>.` index (+1), and
      - hidden_dim from `body.proj_hex.weight` (shape [hidden_dim, F_HEX]).
    Returns (hidden_dim, num_layers); raises KeyError/ValueError if the keys
    aren't present (caller treats that as a bad checkpoint).
    """
    import re

    layer_idxs = set()
    for k in state:
        m = re.match(r"body\.convs\.(\d+)\.", k)
        if m:
            layer_idxs.add(int(m.group(1)))
    if not layer_idxs:
        raise ValueError("no body.convs.* layers found in state_dict")
    num_layers = max(layer_idxs) + 1
    proj = state.get("body.proj_hex.weight")
    if proj is None:
        raise ValueError("missing body.proj_hex.weight in state_dict")
    hidden_dim = int(proj.shape[0])
    return hidden_dim, num_layers


def _load_gnn_model(checkpoint: str, *, hidden_dim=None, num_layers=None, device: str):
    """Load a GnnModel from a .pt checkpoint (handles {'model_state': ...} wrappers).

    When hidden_dim/num_layers are None (the default), the architecture is
    inferred from the checkpoint's state_dict so checkpoints of any trained
    size load without the caller knowing their shape. Explicit values override.
    """
    if not Path(checkpoint).exists():
        raise ValueError(f"checkpoint not found: {checkpoint}")
    import torch
    from catan_gnn.gnn_model import GnnModel
    try:
        obj = torch.load(checkpoint, map_location=device, weights_only=False)
        state = obj["model_state"] if isinstance(obj, dict) and "model_state" in obj else obj
        inferred_dim, inferred_layers = _infer_arch(state)
        h = int(hidden_dim) if hidden_dim is not None else inferred_dim
        n = int(num_layers) if num_layers is not None else inferred_layers
        model = GnnModel(hidden_dim=h, num_layers=n)
        model.load_state_dict(state)
    except Exception as e:
        raise ValueError(f"failed to load checkpoint {checkpoint!r}: {e}") from e
    return model.to(device).eval()


def _build_gnn_bot(spec, *, game, seed):
    checkpoint = spec.get("checkpoint")
    if not checkpoint:
        raise ValueError("GNN bot requires a 'checkpoint' path")
    device = spec.get("device", "cpu")
    # Default to inferring the architecture from the checkpoint; only override
    # if the spec explicitly carries hidden_dim/num_layers.
    hidden_dim = spec.get("hidden_dim")
    num_layers = spec.get("num_layers")
    model = _load_gnn_model(checkpoint, hidden_dim=hidden_dim,
                            num_layers=num_layers, device=device)
    if spec["type"] == "PureGnn":
        from catan_mcts.bots_gnn import PureGnnBot
        return PureGnnBot(model=model, device=device)
    from catan_mcts.experiments.e10e_gnn_mcts import build_gnn_mcts_bot
    sims = int(spec.get("sims", 200))
    return build_gnn_mcts_bot(game, model, sims=sims, seed=seed, device=device)
