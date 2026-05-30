# tests/test_async_mcts.py
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
        assert 0 < int(visits.sum()) <= 15
        legal = set(state.legal_actions())
        assert all(visits[a] == 0 for a in range(ACTION_SPACE_SIZE) if a not in legal)
    finally:
        await ev.stop()


async def test_value_rotated_to_absolute_seat():
    # The GNN value head is ego-relative; _expand_and_evaluate must rotate it to
    # absolute-seat order so backup indexes by node.to_play correctly.
    import numpy as np
    ev = BatchedGnnEvaluator(model=_untrained_model(), device="cpu",
                             max_batch=4, window_ms=5)
    ev.start()
    try:
        mcts = AsyncMcts(evaluator=ev, c=1.4, rng=np.random.default_rng(0))
        state = _leaf_state()
        leaf_mover = state.current_player()
        # Get the raw ego-relative value the evaluator produces for this leaf.
        ego_value, _ = await ev.eval_leaf(state)
        ego_value = np.asarray(ego_value, dtype=np.float32)
        # Drive _expand_and_evaluate on a fresh node for the same state.
        from catan_mcts.async_mcts import Node
        node = Node(state.clone())
        value_abs = await mcts._expand_and_evaluate(node)
        # value_abs[absolute_seat] must equal ego_value[(absolute_seat - leaf_mover) % 4]
        for seat in range(4):
            offset = (seat - leaf_mover) % 4
            assert abs(value_abs[seat] - ego_value[offset]) < 1e-5, (
                f"seat {seat}: abs={value_abs[seat]} != ego[{offset}]={ego_value[offset]}")
    finally:
        await ev.stop()


async def test_play_full_game_terminates_and_records():
    from catan_mcts.async_mcts import play_one_async_game
    ev = BatchedGnnEvaluator(model=_untrained_model(), device="cpu",
                             max_batch=4, window_ms=5)
    ev.start()
    try:
        game = CatanGame(vp_target=10, bonuses=True)
        result = await play_one_async_game(
            game=game, seed=123, evaluator=ev, n_sims=8,
            rng=np.random.default_rng(123), max_steps=200000)
        assert result.terminal is True
        assert -1 <= result.winner <= 3
        assert result.length_in_moves > 0
        assert len(result.moves) > 0
        m = result.moves[0]
        assert m.visit_counts.shape == (ACTION_SPACE_SIZE,)
        assert m.legal_mask.shape == (ACTION_SPACE_SIZE,)
        # final_vp must populate for a finished game (not stay [0,0,0,0]).
        assert len(result.final_vp) == 4
        assert sum(result.final_vp) > 0, f"final_vp not populated: {result.final_vp}"
        # the winner should have the most VP (>= vp_target if a real win)
        if result.winner >= 0:
            assert result.final_vp[result.winner] == max(result.final_vp)
    finally:
        await ev.stop()


async def test_per_game_rng_reproducible():
    from catan_mcts.async_mcts import play_one_async_game
    async def run():
        ev = BatchedGnnEvaluator(model=_untrained_model(), device="cpu",
                                 max_batch=4, window_ms=5)
        ev.start()
        try:
            res = await play_one_async_game(
                game=CatanGame(vp_target=10, bonuses=True), seed=999,
                evaluator=ev, n_sims=8, rng=np.random.default_rng(999),
                max_steps=200000)
            return res.action_history
        finally:
            await ev.stop()
    h1 = await run()
    h2 = await run()
    assert h1 == h2, "same seed produced different play"
