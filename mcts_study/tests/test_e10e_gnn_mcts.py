"""Smoke test for e10e_gnn_mcts experiment wiring.

The GNN+MCTS mechanism itself is covered by test_gnn_evaluator.py
(test_runs_inside_mctsbot_one_full_game). This test covers the e10e-specific
wiring: that build_gnn_mcts_bot produces a working MCTSBot and that the full
4-role seating (PureGnn / GnnMcts / PureGnn / LookV3) constructs and plays a
game end-to-end without error.

Uses an untrained model + tiny sims so it runs fast; we assert wiring, not
strength.
"""
from __future__ import annotations

import numpy as np
import torch

from catan_gnn.gnn_model import GnnModel
from catan_mcts.adapter import CatanGame
from catan_mcts.experiments.e10e_gnn_mcts import build_gnn_mcts_bot


def _untrained_model():
    torch.manual_seed(0)
    return GnnModel(hidden_dim=8, num_layers=2)


def test_build_gnn_mcts_bot_returns_stepping_bot():
    """build_gnn_mcts_bot returns an MCTSBot that picks a legal action."""
    game = CatanGame(vp_target=10, bonuses=True)
    bot = build_gnn_mcts_bot(game, model=_untrained_model(), sims=2,
                             seed=123, device="cpu")
    state = game.new_initial_state(seed=42)
    # Drive to first player decision (skip chance + forced singletons).
    while state.is_chance_node():
        state.apply_action(int(state.chance_outcomes()[0][0]))
    legal = state.legal_actions()
    action = bot.step(state)
    assert int(action) in [int(a) for a in legal]


def test_fresh_evaluator_per_bot_not_shared():
    """Each build_gnn_mcts_bot call must own a fresh GnnEvaluator (id-keyed
    cache), so two bots never share cached forward results across games."""
    game = CatanGame(vp_target=10, bonuses=True)
    m = _untrained_model()
    bot1 = build_gnn_mcts_bot(game, model=m, sims=2, seed=1, device="cpu")
    bot2 = build_gnn_mcts_bot(game, model=m, sims=2, seed=2, device="cpu")
    assert bot1.evaluator is not bot2.evaluator


def test_e10e_main_one_game_per_seating_runs(tmp_path):
    """End-to-end: e10e.main with 1 game/seating, sims=2, untrained models,
    workers=1, CPU. Proves the full PureGnn/GnnMcts/PureGnn/LookV3 lineup
    constructs and plays without error, and writes a run dir."""
    from catan_mcts.experiments.e10e_gnn_mcts import main as e10e_main

    # Write three tiny untrained checkpoints to disk (the experiment loads
    # checkpoints by path).
    ckpts = []
    for i in range(3):
        torch.manual_seed(i)
        model = GnnModel(hidden_dim=8, num_layers=2)
        path = tmp_path / f"ckpt_{i}.pt"
        torch.save({"model_state": model.state_dict()}, path)
        ckpts.append(path)

    out = e10e_main(
        out_root=tmp_path / "runs",
        checkpoint_a=ckpts[0],
        checkpoint_b=ckpts[1],
        checkpoint_c=ckpts[2],
        num_games_per_seating=1,
        gnn_mcts_sims=2,
        base_sims_v3=50,  # LookV3 enforces SIM_FLOOR=50
        lookahead_depth=2,
        hidden_dim=8, num_layers=2,
        vp_target=10, bonuses=True,
        max_seconds=120.0,
        workers=1,
        device="cpu",
    )
    assert out.exists()
    # At least one games parquet should have been written.
    parquets = list(out.rglob("games*.parquet"))
    assert parquets, "no games parquet written"
