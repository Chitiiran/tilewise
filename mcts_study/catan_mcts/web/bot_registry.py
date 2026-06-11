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


# Difficulty ladder. Each tier is justified by a measured winrate (see plan
# 2026-06-10-az-difficulty-bots.md). Checkpoints are stored RELATIVE to the
# server's checkpoints_dir and resolved at build time. There is deliberately
# no mid-sims GnnMcts tier: the 2026-06-01 cheap-search sweep showed sims
# 8/16/32 play WORSE than raw argmax (the "valley") — only ~200-sim search
# beats LookaheadV3.
_CELL6_CKPT = ("training/loss_aug/06_cand11_cand8_cand10_h128_l4/"
               "training_h128_l4/checkpoint_epoch10.pt")

_DIFFICULTIES: list[dict] = [
    {"id": "beginner", "label": "Beginner (random)",
     "spec": {"type": "Random"}},
    {"id": "easy", "label": "Easy (greedy)",
     "spec": {"type": "Greedy"}},
    {"id": "medium", "label": "Medium (GNN policy)",
     "spec": {"type": "PureGnn", "checkpoint_rel": _CELL6_CKPT}},
    {"id": "hard", "label": "Hard (lookahead search)",
     "spec": {"type": "LookaheadMctsV3"}},
    {"id": "expert", "label": "Expert (GNN + deep search)",
     "spec": {"type": "GnnMcts", "checkpoint_rel": _CELL6_CKPT,
              "sims": 200, "device": "cpu"}},
]


def list_difficulties() -> list[dict]:
    """Difficulty presets for the lobby, in ascending strength order."""
    return [dict(d, spec=dict(d["spec"])) for d in _DIFFICULTIES]


def resolve_seat_spec(spec: dict, *, checkpoints_dir) -> dict:
    """Turn a lobby seat spec into a buildable bot spec.

    Accepts either {"type": ...} (passed through unchanged, back-compat) or
    {"difficulty": "<id>"} (expanded from the preset table). Preset specs
    carrying `checkpoint_rel` get it resolved against checkpoints_dir into an
    absolute `checkpoint`; a missing file is a ValueError naming the expected
    relative path so deployment gaps are loud, not mysterious.
    """
    if spec.get("type"):
        return spec
    diff_id = spec.get("difficulty")
    if not diff_id:
        raise ValueError("seat spec needs a 'type' or a 'difficulty'")
    preset = next((d for d in _DIFFICULTIES if d["id"] == diff_id), None)
    if preset is None:
        known = ", ".join(d["id"] for d in _DIFFICULTIES)
        raise ValueError(f"unknown difficulty {diff_id!r} (known: {known})")
    out = dict(preset["spec"])
    rel = out.pop("checkpoint_rel", None)
    if rel is not None:
        if checkpoints_dir is None:
            raise ValueError(f"difficulty {diff_id!r} needs a checkpoint, "
                             f"but no checkpoints_dir is configured")
        path = Path(checkpoints_dir) / rel
        if not path.exists():
            raise ValueError(f"difficulty {diff_id!r} checkpoint missing: "
                             f"expected {rel} under {checkpoints_dir}")
        out["checkpoint"] = str(path)
    return out


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
    """Recursively list *.pt files under `checkpoints_dir`.

    Many runs reuse the same filename (e.g. dozens of `checkpoint_best.pt`),
    so the bare filename is not enough to tell them apart in the lobby. Each
    entry carries:
      - name:  the bare filename (kept for back-compat),
      - path:  absolute path (what build() loads),
      - label: the path RELATIVE to the checkpoints root (POSIX slashes) — this
               disambiguates same-named files across run dirs, and
      - dir:   the relative parent dir ("" for a top-level file) — used to
               group the dropdown by run via <optgroup>.
    Sorted by label so checkpoints from the same run cluster together.
    """
    root = Path(checkpoints_dir)
    if not root.exists():
        return []
    out = []
    for p in sorted(root.rglob("*.pt")):
        rel = p.relative_to(root)
        label = rel.as_posix()
        parent = rel.parent.as_posix()
        out.append({
            "name": p.name,
            "path": str(p),
            "label": label,
            "dir": "" if parent == "." else parent,
        })
    out.sort(key=lambda c: c["label"])
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
