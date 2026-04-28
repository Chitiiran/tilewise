"""OpenSpiel adapter wrapping catan_bot._engine.Engine.

CatanGame is registered as a real `pyspiel.Game` subclass (not duck-typed), so
that OpenSpiel's `MCTSBot` can call `game.get_type()` / `game.max_utility()`
during its `__init__` contract check.

Naming convention: `pyspiel.State`'s public methods (`apply_action`,
`legal_actions`, `action_to_string`) dispatch into protected hooks
(`_apply_action`, `_legal_actions`, `_action_to_string`) for Python subclasses.
We override the protected hooks. Public methods we override directly:
`current_player`, `is_terminal`, `is_chance_node`, `chance_outcomes`, `returns`.
"""
from __future__ import annotations

import json

import pyspiel
from catan_bot import _engine

from . import ACTION_SPACE_SIZE

NUM_PLAYERS = 4

_GAME_TYPE = pyspiel.GameType(
    short_name="catan_v1",
    long_name="Catan (Tier-1, fixed board)",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.GENERAL_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=NUM_PLAYERS,
    min_num_players=NUM_PLAYERS,
    provides_information_state_string=False,
    provides_information_state_tensor=False,
    provides_observation_string=False,
    provides_observation_tensor=False,
    parameter_specification={"seed": 0},
)

# Roll: 11 outcomes (dice sums 2..12). Steal: bounded by victim hand. Tier-1
# players cannot accumulate more than ~30 cards before being forced to discard.
# 64 leaves comfortable headroom on both axes.
_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=ACTION_SPACE_SIZE,
    max_chance_outcomes=64,
    num_players=NUM_PLAYERS,
    min_utility=-1.0,
    max_utility=1.0,
    utility_sum=None,
    max_game_length=5000,
)


class CatanGame(pyspiel.Game):
    """OpenSpiel-registered Catan game."""

    def __init__(self, params=None):
        params = params or {}
        super().__init__(_GAME_TYPE, _GAME_INFO, params)
        self._default_seed = int(params.get("seed", 0))

    def new_initial_state(self, seed: int | None = None) -> "CatanState":
        s = self._default_seed if seed is None else int(seed)
        return CatanState(self, _engine.Engine(s), initial_seed=s)

    def make_py_observer(self, iig_obs_type=None, params=None):
        # We don't expose observation tensors yet — MCTS doesn't need them.
        return None

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
        # Rebuild a state attached to a fresh game instance — callers using the
        # static method don't have a CatanGame handy and that's fine for MCTS.
        return CatanState(CatanGame(), engine, initial_seed=seed)


class CatanState(pyspiel.State):
    """OpenSpiel state wrapping `catan_bot._engine.Engine`."""

    def __init__(self, game, engine, initial_seed: int = 0) -> None:
        super().__init__(game)
        self._engine = engine
        self._initial_seed = initial_seed

    # --- Public methods OpenSpiel queries directly ----------------------------

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
        return [(int(v), float(p)) for v, p in self._engine.chance_outcomes()]

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

    # --- Protected hooks (called by pyspiel.State public dispatchers) --------

    def _legal_actions(self, player) -> list[int]:
        # Sequential game — engine knows whose turn it is. Ignore `player`.
        return [int(a) for a in self._engine.legal_actions()]

    def _apply_action(self, action) -> None:
        action = int(action)
        if self._engine.is_chance_pending():
            self._engine.apply_chance_outcome(action)
        else:
            self._engine.step(action)

    def _action_to_string(self, player, action) -> str:
        return f"a{int(action)}"

    # --- Hash / debug --------------------------------------------------------

    def __str__(self) -> str:
        return f"CatanState(seed={self._initial_seed},history={self._engine.action_history()})"

    # --- Clone ---------------------------------------------------------------
    # pyspiel.State's `clone()` is C++ and doesn't know about our `_engine`
    # attribute; override to copy the engine deterministically.

    def clone(self) -> "CatanState":
        new = CatanState.__new__(CatanState)
        pyspiel.State.__init__(new, self.get_game())
        new._engine = self._engine.clone()
        new._initial_seed = self._initial_seed
        return new

    # --- Our own helpers (not OpenSpiel hooks) -------------------------------

    def history(self) -> list[int]:
        return [int(x) for x in self._engine.action_history()]

    def serialize(self) -> str:
        return json.dumps({"seed": self._initial_seed, "history": self.history()})


# Register the game with the OpenSpiel library so `pyspiel.load_game("catan_v1")`
# works and the C++ side knows our type.
pyspiel.register_game(_GAME_TYPE, CatanGame)
