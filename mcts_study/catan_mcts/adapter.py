"""OpenSpiel adapter wrapping catan_bot._engine.Engine."""
from __future__ import annotations

from typing import Iterable

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
            return -1  # OpenSpiel's TERMINAL marker convention
        if self._engine.is_chance_pending():
            return -2  # OpenSpiel's CHANCE_PLAYER_ID is -1; we'll align in Task 3.
        return int(self._engine.current_player())

    def legal_actions(self) -> list[int]:
        return [int(a) for a in self._engine.legal_actions()]

    def apply_action(self, action: int) -> None:
        self._engine.step(int(action))
