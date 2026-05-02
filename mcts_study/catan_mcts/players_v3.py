"""v3 self-play players.

v3.4 (per spec §5): the data-generating player is `LookaheadMctsV3` — an
MCTSBot whose simulation budget is reshaped per move based on the acting
player's current VP. The intuition: when a player has 0-1 VP they have many
plausible build paths and search pays off; when they have 4 VP (one shy of
winning under v3) the choice space collapses and ~50 sims is enough.

Schedule (locked in spec §5):
    sims = max(50, round(200 * 0.7 ** acting_vp))

acting_vp | sims
   0      | 200
   1      | 140
   2      | 98
   3      | 69
   4      | 50

We do NOT vary the sim budget on chance nodes or non-decision branches —
those are handled by MCTSBot internally. The sim count only matters at the
root, where MCTSBot iterates `max_simulations` times per `step()` call.

Implementation note: rather than rebuild MCTSBot every move, we subclass it
and mutate `self.max_simulations` in `step()` before delegating to super().
This keeps OpenSpiel's UCT / tree code untouched.
"""
from __future__ import annotations

import numpy as np
from open_spiel.python.algorithms import mcts as os_mcts

from .evaluator import LookaheadVpEvaluator


# Schedule constants — kept as module-level so tests and journals can reference them.
SIM_BASE = 200
SIM_DECAY = 0.7
SIM_FLOOR = 50


def sims_for_player(acting_vp: int) -> int:
    """v3 per-player VP-aware exponential sim schedule.

    Args:
        acting_vp: The current player's VP at the root of the search.
    Returns:
        Number of MCTS simulations to run for this move.
    """
    return max(SIM_FLOOR, round(SIM_BASE * (SIM_DECAY ** int(acting_vp))))


class _AdaptiveSimsMCTSBot(os_mcts.MCTSBot):
    """MCTSBot variant that reshapes `max_simulations` per call.

    Reads the acting player's VP from `state._engine.vp(acting_player)` at
    the moment of `step()` or `mcts_search()`, computes the sim budget via
    `sims_for_player`, sets `self.max_simulations`, and delegates to
    MCTSBot.

    Both step() and mcts_search() are overridden — the recorder uses
    mcts_search() directly to extract visit counts, so we have to apply
    the schedule there too or recorded games would burn the base sim
    budget on every move regardless of VP.

    Falls back to a constant base_sims if the wrapped state doesn't expose
    a `_engine` (defensive — happens in synthetic tests).
    """

    def __init__(self, *args, base_sims: int = SIM_BASE, **kwargs):
        # Initialize with base_sims; step() / mcts_search() will override per call.
        kwargs["max_simulations"] = base_sims
        super().__init__(*args, **kwargs)
        self._base_sims = int(base_sims)

    def _set_sims_for_state(self, state) -> None:
        engine = getattr(state, "_engine", None)
        if engine is None:
            self.max_simulations = self._base_sims
            return
        try:
            acting = state.current_player()
            if 0 <= acting <= 3:
                self.max_simulations = sims_for_player(engine.vp(acting))
            else:
                self.max_simulations = self._base_sims
        except Exception:
            self.max_simulations = self._base_sims

    def step(self, state):
        self._set_sims_for_state(state)
        return super().step(state)

    def mcts_search(self, state):
        self._set_sims_for_state(state)
        return super().mcts_search(state)


def build_lookahead_mcts_v3(game, lookahead_depth: int = 10, seed: int = 0,
                             base_sims: int = SIM_BASE):
    """Construct a v3 LookaheadMctsV3 player.

    Args:
        game: A CatanGame (typically constructed with vp_target=5, bonuses=False).
        lookahead_depth: Greedy lookahead depth for the leaf evaluator.
        seed: Base seed for evaluator randomness.
        base_sims: Sim budget at acting_vp=0. Must satisfy base_sims >= SIM_FLOOR.

    Returns:
        An MCTSBot subclass whose simulation count adapts to the acting
        player's current VP.
    """
    if base_sims < SIM_FLOOR:
        raise ValueError(f"base_sims={base_sims} below SIM_FLOOR={SIM_FLOOR}")
    rng = np.random.default_rng(seed)
    evaluator = LookaheadVpEvaluator(depth=lookahead_depth, base_seed=seed)
    return _AdaptiveSimsMCTSBot(
        game=game, uct_c=1.4, base_sims=base_sims,
        evaluator=evaluator, solve=False, random_state=rng,
    )
