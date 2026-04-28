"""OpenSpiel adapter wrapping catan_bot._engine.Engine."""
from __future__ import annotations

from typing import Iterable

import pyspiel
from catan_bot import _engine

from . import ACTION_SPACE_SIZE

NUM_PLAYERS = 4


class CatanGame:
    """OpenSpiel-style Game for our Catan engine.

    Note: We don't subclass `pyspiel.Game` directly — pyspiel.Game requires C++-side
    registration. Instead we expose the duck-typed interface OpenSpiel's Python MCTSBot
    needs. If a future algorithm requires registered pyspiel games, we'll add a
    register-on-import path; until then, duck typing is simpler and works for MCTS.
    """

    def __init__(self, default_seed: int = 0) -> None:
        self._default_seed = default_seed

    def num_distinct_actions(self) -> int:
        return ACTION_SPACE_SIZE

    def num_players(self) -> int:
        return NUM_PLAYERS

    def new_initial_state(self, seed: int | None = None) -> "CatanState":
        s = self._default_seed if seed is None else seed
        return CatanState(_engine.Engine(s))


class CatanState:
    def __init__(self, engine) -> None:  # engine: catan_bot._engine.Engine
        self._engine = engine

    def is_terminal(self) -> bool:
        return self._engine.is_terminal()

    def current_player(self) -> int:
        if self.is_terminal():
            return pyspiel.PlayerId.TERMINAL
        if self._engine.is_chance_pending():
            return pyspiel.PlayerId.CHANCE
        return int(self._engine.current_player())

    def is_chance_node(self) -> bool:
        return self._engine.is_chance_pending()

    def chance_outcomes(self) -> list[tuple[int, float]]:
        # PyO3 returns list[(u32, f64)]; ensure Python ints/floats.
        return [(int(v), float(p)) for v, p in self._engine.chance_outcomes()]

    def legal_actions(self) -> list[int]:
        return [int(a) for a in self._engine.legal_actions()]

    def apply_action(self, action: int) -> None:
        action = int(action)
        if self._engine.is_chance_pending():
            self._engine.apply_chance_outcome(action)
        else:
            self._engine.step(action)
