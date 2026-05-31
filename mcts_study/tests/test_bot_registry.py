"""Tests for bot discovery + construction."""
from __future__ import annotations

import pytest


def test_list_types_includes_core_bots():
    from catan_mcts.web import bot_registry
    types = {t["id"] for t in bot_registry.list_types()}
    assert {"Random", "Greedy", "LookaheadMctsV3", "PureGnn", "GnnMcts"} <= types


def test_build_random_bot():
    from catan_mcts.web import bot_registry
    from catan_mcts.adapter import CatanGame
    game = CatanGame()
    bot = bot_registry.build({"type": "Random"}, game=game, seed=7)
    state = game.new_initial_state(seed=7)
    action = bot.step(state)
    assert action in state.legal_actions()


def test_build_greedy_bot():
    from catan_mcts.web import bot_registry
    from catan_mcts.adapter import CatanGame
    game = CatanGame()
    bot = bot_registry.build({"type": "Greedy"}, game=game, seed=1)
    state = game.new_initial_state(seed=1)
    assert bot.step(state) in state.legal_actions()


def test_build_unknown_type_raises():
    from catan_mcts.web import bot_registry
    from catan_mcts.adapter import CatanGame
    with pytest.raises(ValueError, match="unknown bot type"):
        bot_registry.build({"type": "Nope"}, game=CatanGame(), seed=0)


def test_list_checkpoints_scans_dir(tmp_path):
    from catan_mcts.web import bot_registry
    (tmp_path / "a.pt").write_bytes(b"x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.pt").write_bytes(b"y")
    (tmp_path / "notes.txt").write_text("ignore me")
    cps = bot_registry.list_checkpoints(tmp_path)
    names = {c["name"] for c in cps}
    assert "a.pt" in names and "b.pt" in names
    assert all(c["path"].endswith(".pt") for c in cps)
    assert not any(c["name"] == "notes.txt" for c in cps)


def test_list_checkpoints_label_disambiguates_by_dir(tmp_path):
    """Same filename in different run dirs must get distinct labels + a dir tag."""
    from catan_mcts.web import bot_registry
    (tmp_path / "runA").mkdir()
    (tmp_path / "runB").mkdir()
    (tmp_path / "runA" / "checkpoint_best.pt").write_bytes(b"x")
    (tmp_path / "runB" / "checkpoint_best.pt").write_bytes(b"y")
    (tmp_path / "top.pt").write_bytes(b"z")
    cps = bot_registry.list_checkpoints(tmp_path)
    labels = {c["label"] for c in cps}
    # The two same-named checkpoints are distinguishable by their dir-qualified label.
    assert "runA/checkpoint_best.pt" in labels
    assert "runB/checkpoint_best.pt" in labels
    # A top-level checkpoint's label is just its filename; its dir tag is "" (root).
    top = next(c for c in cps if c["name"] == "top.pt")
    assert top["label"] == "top.pt"
    assert top["dir"] == ""
    # Each entry carries a `dir` group tag for the dropdown's optgroup.
    a = next(c for c in cps if c["label"] == "runA/checkpoint_best.pt")
    assert a["dir"] == "runA"


def test_build_gnn_bad_checkpoint_raises(tmp_path):
    from catan_mcts.web import bot_registry
    from catan_mcts.adapter import CatanGame
    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"not a torch checkpoint")
    with pytest.raises(ValueError, match="checkpoint"):
        bot_registry.build(
            {"type": "PureGnn", "checkpoint": str(bad)},
            game=CatanGame(), seed=0,
        )


def test_build_gnn_missing_checkpoint_raises(tmp_path):
    from catan_mcts.web import bot_registry
    from catan_mcts.adapter import CatanGame
    with pytest.raises(ValueError, match="checkpoint"):
        bot_registry.build(
            {"type": "PureGnn", "checkpoint": str(tmp_path / "nope.pt")},
            game=CatanGame(), seed=0,
        )


def test_infer_arch_reads_layers_and_hidden_dim():
    """Architecture (hidden_dim, num_layers) is inferred from a state_dict."""
    import torch
    from catan_gnn.gnn_model import GnnModel
    from catan_mcts.web import bot_registry
    model = GnnModel(hidden_dim=64, num_layers=3)
    hidden_dim, num_layers = bot_registry._infer_arch(model.state_dict())
    assert hidden_dim == 64
    assert num_layers == 3


def test_build_pure_gnn_loads_nondefault_arch(tmp_path):
    """A checkpoint trained with non-default hidden_dim/num_layers loads
    without the caller specifying the architecture."""
    import torch
    from catan_gnn.gnn_model import GnnModel
    from catan_mcts.web import bot_registry
    from catan_mcts.adapter import CatanGame
    # Save a model whose architecture differs from the old hardcoded 32/2.
    model = GnnModel(hidden_dim=64, num_layers=3)
    ckpt = tmp_path / "h64l3.pt"
    torch.save(model.state_dict(), ckpt)
    bot = bot_registry.build(
        {"type": "PureGnn", "checkpoint": str(ckpt)},
        game=CatanGame(), seed=0,
    )
    # It built a usable PureGnnBot (has a .step) without an arch in the spec.
    assert hasattr(bot, "step")
