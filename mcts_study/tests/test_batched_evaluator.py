# tests/test_batched_evaluator.py
import asyncio
import random
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


async def test_batch_fills_to_max_in_one_forward():
    ev = BatchedGnnEvaluator(model=_untrained_model(), device="cpu",
                             max_batch=8, window_ms=50)
    ev.start()
    try:
        states = [_leaf_state() for _ in range(8)]
        results = await asyncio.gather(*[ev.eval(s) for s in states])
        assert len(results) == 8
        # 8 requests, max_batch 8 -> exactly one batch.
        assert ev.total_batches == 1
    finally:
        await ev.stop()


async def test_window_fires_partial_batch():
    # 3 requests < max_batch 8: must still resolve via the time window, not hang.
    ev = BatchedGnnEvaluator(model=_untrained_model(), device="cpu",
                             max_batch=8, window_ms=20)
    ev.start()
    try:
        states = [_leaf_state() for _ in range(3)]
        results = await asyncio.wait_for(
            asyncio.gather(*[ev.eval(s) for s in states]), timeout=2.0)
        assert len(results) == 3
    finally:
        await ev.stop()


async def test_all_parked_flushes_immediately():
    # When pending >= active_game_count, flush without waiting for the window.
    ev = BatchedGnnEvaluator(model=_untrained_model(), device="cpu",
                             max_batch=64, window_ms=10_000)  # huge window
    ev.active_game_count = 3
    ev.start()
    try:
        states = [_leaf_state() for _ in range(3)]
        # If the all-parked clause works, this resolves well under the 10s window.
        results = await asyncio.wait_for(
            asyncio.gather(*[ev.eval(s) for s in states]), timeout=2.0)
        assert len(results) == 3
    finally:
        await ev.stop()


async def test_eval_leaf_helper_skips_model_for_terminal():
    # eval_leaf() returns state.returns() for terminals WITHOUT enqueuing a GPU request.
    ev = BatchedGnnEvaluator(model=_untrained_model(), device="cpu",
                             max_batch=8, window_ms=5)
    ev.start()
    try:
        game = CatanGame(vp_target=10, bonuses=True)
        # Drive a game to terminal with a seeded random policy (always-first loops forever).
        rng = random.Random(7)
        state = game.new_initial_state(seed=7)
        steps = 0
        while not state.is_terminal() and steps < 200000:
            if state.is_chance_node():
                state.apply_action(int(state.chance_outcomes()[0][0]))
            else:
                state.apply_action(int(rng.choice(state.legal_actions())))
            steps += 1
        assert state.is_terminal(), f"game did not terminate after {steps} steps"
        before = ev.total_requests
        value, priors = await ev.eval_leaf(state)
        assert ev.total_requests == before  # no GPU request enqueued
        assert priors is None
        assert list(value) == state.returns()
    finally:
        await ev.stop()


async def test_eval_leaf_non_terminal_returns_normalized_priors():
    ev = BatchedGnnEvaluator(model=_untrained_model(), device="cpu",
                             max_batch=8, window_ms=5)
    ev.start()
    try:
        state = _leaf_state()
        assert not state.is_terminal()
        value, priors = await ev.eval_leaf(state)
        assert value.shape == (4,)
        assert priors is not None and len(priors) > 0
        legal = state.legal_actions()
        assert [a for a, _ in priors] == legal
        assert abs(sum(p for _, p in priors) - 1.0) < 1e-5
    finally:
        await ev.stop()
