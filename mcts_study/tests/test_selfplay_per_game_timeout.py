"""Per-game wall-clock timeout in self-play (2026-06-15).

Pathologically long self-play games (heading toward the 200k-step cap) have no
per-game wall-clock cap — only the 9h per-WORKER cap. So one long game stalls
its whole worker (24 concurrent games finalize together), which stalls the whole
self-play stage, which needed a manual straggler-kill in production iter_6.

Fix: play_one_async_game takes a deadline; when wall-clock exceeds it the game
loop stops and returns a NON-terminal GameResult (partial action_history/moves),
which _play_and_record records as timed_out=True — preserving the
strategically-interesting long game's data instead of dropping it.
"""
from __future__ import annotations

import numpy as np
import torch


def _untrained_model():
    torch.manual_seed(0)
    from catan_gnn.gnn_model import GnnModel
    return GnnModel(hidden_dim=8, num_layers=2)


async def test_play_one_game_stops_at_deadline():
    """With a deadline that the injected clock blows past immediately, the game
    returns non-terminal (timed out) rather than running to a natural end."""
    from catan_mcts.async_mcts import play_one_async_game
    from catan_mcts.batched_evaluator import BatchedGnnEvaluator
    from catan_mcts.adapter import CatanGame

    ev = BatchedGnnEvaluator(model=_untrained_model(), device="cpu",
                             max_batch=4, window_ms=5)
    ev.start()
    try:
        game = CatanGame(vp_target=10, bonuses=True)
        # clock: stays under the 100s deadline for the first ~50 loop-checks
        # (the game makes real moves), then jumps past it so the game stops
        # MID-play (non-terminal) — the real straggler scenario.
        seq = [float(i) for i in range(50)] + [1_000.0] * 100_000
        ticks = iter(seq)
        clock = lambda: next(ticks)
        result = await play_one_async_game(
            game=game, seed=7, evaluator=ev, n_sims=4,
            rng=np.random.default_rng(7), self_play=True,
            deadline_seconds=100.0, clock=clock)
        assert result.terminal is False        # stopped by the deadline
        assert result.length_in_moves > 0      # made progress before the cut
    finally:
        await ev.stop()


async def test_no_deadline_plays_to_terminal():
    """Default (no deadline) is unchanged: a small game plays to a natural end."""
    from catan_mcts.async_mcts import play_one_async_game
    from catan_mcts.batched_evaluator import BatchedGnnEvaluator
    from catan_mcts.adapter import CatanGame

    ev = BatchedGnnEvaluator(model=_untrained_model(), device="cpu",
                             max_batch=4, window_ms=5)
    ev.start()
    try:
        game = CatanGame(vp_target=10, bonuses=True)
        result = await play_one_async_game(
            game=game, seed=3, evaluator=ev, n_sims=4,
            rng=np.random.default_rng(3), self_play=True)
        # vp_target=10 full game terminates well under the 200k step cap
        assert result.terminal is True
    finally:
        await ev.stop()
