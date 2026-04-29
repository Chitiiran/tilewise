"""Shared experiment infrastructure: run-directory creation, the canonical game loop."""
from __future__ import annotations

import datetime as dt
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .. import ACTION_SPACE_SIZE


@dataclass
class GameOutcome:
    seed: int
    winner: int
    final_vp: list[int]
    length_in_moves: int
    timed_out: bool = False  # v2: True when max_seconds aborted the game


def make_run_dir(parent: Path, name: str) -> Path:
    ts = dt.datetime.utcnow().strftime("%Y-%m-%dT%H-%M")
    d = parent / f"{ts}-{name}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _sample_chance_outcome(state, rng: random.Random) -> int:
    outcomes = state.chance_outcomes()
    r = rng.random()
    cum = 0.0
    for v, p in outcomes:
        cum += p
        if r <= cum:
            return int(v)
    return int(outcomes[-1][0])


def play_one_game(
    *,
    game,
    bots: dict,
    seed: int,
    chance_rng: random.Random,
    recorded_player: Optional[int] = None,
    recorder_game=None,    # an active _GameRecorder context
    mcts_bot=None,         # only required if recording (visit count extraction)
    max_steps: int = 30000,
    max_seconds: Optional[float] = None,
) -> GameOutcome:
    """Drive one game to completion. Optionally record `recorded_player`'s MCTS moves.

    For an MCTS player, `bots[recorded_player]` should be the OpenSpiel `MCTSBot`,
    AND `mcts_bot` must point at the same instance — the recorder needs to call
    `mcts_bot.mcts_search(state)` to extract visit counts.

    max_steps default 30000 because Tier-1 chance-aware games take ~12-15k steps.
    max_seconds is a v2 wall-clock cap; on timeout returns winner=-1, timed_out=True.
    """
    from ..recorder import visit_counts_from_root

    state = game.new_initial_state(seed=seed)
    move_index = 0
    steps = 0
    t_start = time.perf_counter()
    timed_out = False

    while not state.is_terminal() and steps < max_steps:
        if max_seconds is not None and time.perf_counter() - t_start > max_seconds:
            timed_out = True
            break
        if state.is_chance_node():
            state.apply_action(_sample_chance_outcome(state, chance_rng))
        else:
            p = state.current_player()
            bot = bots[p]
            if recorder_game is not None and p == recorded_player and mcts_bot is not None:
                # Search once to populate the tree; record visit counts; play best.
                root = mcts_bot.mcts_search(state)
                visits = visit_counts_from_root(root)
                legal = state.legal_actions()
                best = max(legal, key=lambda a: int(visits[a]))
                mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
                mask[legal] = True
                root_value = float(getattr(root, "total_reward", 0.0)) / max(
                    1, int(getattr(root, "explore_count", 1))
                )
                recorder_game.record_move(
                    current_player=p,
                    move_index=move_index,
                    legal_action_mask=mask,
                    mcts_visit_counts=visits,
                    action_taken=int(best),
                    mcts_root_value=root_value,
                )
                move_index += 1
                state.apply_action(int(best))
            else:
                action = bot.step(state)
                state.apply_action(int(action))
        steps += 1

    winner = -1
    final_vp = [0, 0, 0, 0]
    if state.is_terminal():
        rs = state.returns()
        if 1.0 in rs:
            winner = rs.index(1.0)
        # Pull VP from engine stats. The adapter exposes the underlying engine via _engine.
        stats = state._engine.stats()
        final_vp = [int(stats["players"][p]["vp_final"]) for p in range(4)]
    return GameOutcome(
        seed=seed, winner=winner, final_vp=final_vp,
        length_in_moves=steps, timed_out=timed_out,
    )
