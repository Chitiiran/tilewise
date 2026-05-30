# tests/test_batched_evaluator.py
import asyncio
import numpy as np
import torch
from catan_gnn.gnn_model import GnnModel
from catan_mcts.adapter import CatanGame
from catan_mcts.batched_evaluator import BatchedGnnEvaluator
from catan_mcts import ACTION_SPACE_SIZE


def _untrained_model():
    torch.manual_seed(0)
    return GnnModel(hidden_dim=8, num_layers=2)


def _leaf_state():
    game = CatanGame(vp_target=10, bonuses=True)
    state = game.new_initial_state(seed=42)
    while state.is_chance_node():
        state.apply_action(int(state.chance_outcomes()[0][0]))
    return state


async def test_single_eval_returns_value_and_policy():
    ev = BatchedGnnEvaluator(model=_untrained_model(), device="cpu",
                             max_batch=8, window_ms=5)
    ev.start()
    try:
        value, policy = await ev.eval(_leaf_state())
        assert isinstance(value, np.ndarray) and value.shape == (4,)
        assert isinstance(policy, np.ndarray)
        assert policy.shape == (ACTION_SPACE_SIZE,)
        assert np.isfinite(value).all()
    finally:
        await ev.stop()
