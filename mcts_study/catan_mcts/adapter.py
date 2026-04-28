"""OpenSpiel adapter wrapping catan_bot._engine.Engine."""
from __future__ import annotations

import json
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
        return CatanState(_engine.Engine(s), initial_seed=s)

    @staticmethod
    def deserialize(blob: str) -> "CatanState":
        data = json.loads(blob)
        seed = int(data["seed"])
        history = [int(x) for x in data["history"]]
        engine = _engine.Engine(seed)
        CHANCE_FLAG = 0x80000000
        for action_id in history:
            if action_id & CHANCE_FLAG:
                engine.apply_chance_outcome(action_id & ~CHANCE_FLAG)
            else:
                engine.step(action_id)
        return CatanState(engine, initial_seed=seed)


class CatanState:
    def __init__(self, engine, initial_seed: int = 0) -> None:  # engine: catan_bot._engine.Engine
        self._engine = engine
        self._initial_seed = initial_seed

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

    def clone(self) -> "CatanState":
        return CatanState(self._engine.clone(), initial_seed=self._initial_seed)

    def history(self) -> list[int]:
        return [int(x) for x in self._engine.action_history()]

    def returns(self) -> list[float]:
        if not self.is_terminal():
            return [0.0] * NUM_PLAYERS
        stats = self._engine.stats()
        winner = int(stats["winner_player_id"])
        if winner < 0:
            return [0.0] * NUM_PLAYERS
        out = [-1.0] * NUM_PLAYERS
        out[winner] = 1.0
        return out

    def serialize(self) -> str:
        # Stable, unique, deterministic: (seed, action_history) reconstructs everything.
        return json.dumps({"seed": self._initial_seed, "history": self.history()})
