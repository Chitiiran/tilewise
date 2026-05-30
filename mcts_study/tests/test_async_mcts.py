# tests/test_async_mcts.py
import asyncio
import numpy as np
import torch
from catan_gnn.gnn_model import GnnModel
from catan_mcts.adapter import CatanGame
from catan_mcts.batched_evaluator import BatchedGnnEvaluator
from catan_mcts.async_mcts import AsyncMcts
from catan_mcts import ACTION_SPACE_SIZE


def _untrained_model():
    torch.manual_seed(0)
    return GnnModel(hidden_dim=8, num_layers=2)


def _leaf_state(seed=42):
    game = CatanGame(vp_target=10, bonuses=True)
    state = game.new_initial_state(seed=seed)
    while state.is_chance_node():
        state.apply_action(int(state.chance_outcomes()[0][0]))
    return state


async def test_search_returns_visit_counts():
    ev = BatchedGnnEvaluator(model=_untrained_model(), device="cpu",
                             max_batch=4, window_ms=5)
    ev.start()
    try:
        mcts = AsyncMcts(evaluator=ev, c=1.4, rng=np.random.default_rng(0))
        state = _leaf_state()
        visits = await mcts.search(state, n_sims=16)
        assert visits.shape == (ACTION_SPACE_SIZE,)
        assert 0 < int(visits.sum()) <= 16
        legal = set(state.legal_actions())
        assert all(visits[a] == 0 for a in range(ACTION_SPACE_SIZE) if a not in legal)
    finally:
        await ev.stop()
