"""Tests for ValueQGnnBot — 1-ply value-Q deployment of the GNN.

The diagnosis (2026-06-01-puregnn-plateau-diagnosis.md) showed PureGnn's
argmax-of-policy discards the value information that distinguishes Catan's
frequently-near-equal moves. ValueQGnnBot instead scores each legal action by
the VALUE head of its resulting child (the head D1 proved fits well) and picks
the action that maximises the MOVER's value. No tree, one batched eval per
legal set.
"""
import numpy as np
import torch

from catan_gnn.gnn_model import GnnModel
from catan_mcts.adapter import CatanGame
from catan_mcts.batched_evaluator import BatchedGnnEvaluator


def _untrained_model():
    torch.manual_seed(0)
    return GnnModel(hidden_dim=8, num_layers=2)


def _leaf_state(seed=42):
    game = CatanGame(vp_target=10, bonuses=True)
    state = game.new_initial_state(seed=seed)
    while state.is_chance_node():
        state.apply_action(int(state.chance_outcomes()[0][0]))
    return state


async def test_single_legal_action_returned_without_eval():
    # Forced move: return the only legal action, no model calls needed.
    from catan_mcts.value_q_bot import ValueQGnnBot
    ev = BatchedGnnEvaluator(model=_untrained_model(), device="cpu",
                             max_batch=4, window_ms=5)
    ev.start()
    try:
        bot = ValueQGnnBot(evaluator=ev)

        class OneActionState:
            def legal_actions(self):
                return [7]
        assert await bot.step(OneActionState()) == 7
    finally:
        await ev.stop()


async def test_picks_action_maximising_movers_child_value():
    # The bot must pick, over all legal actions a, the action whose child state
    # gives the HIGHEST value to the current mover (rotating the child's
    # ego-relative value into the parent mover's seat).
    from catan_mcts.value_q_bot import ValueQGnnBot
    ev = BatchedGnnEvaluator(model=_untrained_model(), device="cpu",
                             max_batch=4, window_ms=5)
    ev.start()
    try:
        bot = ValueQGnnBot(evaluator=ev)
        state = _leaf_state()
        parent_mover = state.current_player()
        legal = list(state.legal_actions())
        assert len(legal) > 1

        # Independently compute mover-value for each legal action the same way
        # the bot should, and confirm the bot returns the argmax.
        scores = {}
        for a in legal:
            child = state.clone()
            child.apply_action(int(a))
            if child.is_terminal():
                val = np.asarray(child.returns(), dtype=np.float32)
                scores[a] = float(val[parent_mover])  # returns() is absolute-seat
            else:
                child_mover = child.current_player()
                ego, _ = await ev.eval_leaf(child)
                ego = np.asarray(ego, dtype=np.float32)
                scores[a] = float(ego[(parent_mover - child_mover) % 4])
        expected = max(scores, key=lambda a: scores[a])

        chosen = await bot.step(state)
        assert chosen in legal
        assert chosen == expected, (
            f"bot chose {chosen} (score {scores[chosen]:.4f}) but "
            f"max-mover-value is {expected} (score {scores[expected]:.4f})")
    finally:
        await ev.stop()


async def test_step_is_deterministic():
    # No exploration: same state -> same action every time.
    from catan_mcts.value_q_bot import ValueQGnnBot
    ev = BatchedGnnEvaluator(model=_untrained_model(), device="cpu",
                             max_batch=4, window_ms=5)
    ev.start()
    try:
        bot = ValueQGnnBot(evaluator=ev)
        state = _leaf_state(seed=7)
        a1 = await bot.step(state.clone())
        a2 = await bot.step(state.clone())
        assert a1 == a2
    finally:
        await ev.stop()
