"""Gym-style wrapper around the Rust Engine."""
from __future__ import annotations
import numpy as np
from catan_bot._engine import Engine as _Engine, action_space_size

ACTION_SPACE_SIZE = action_space_size()


class CatanEnv:
    """Single-game Catan environment.

    `step(action_id)` returns `(obs, reward, done, info)` where:
      - obs is a dict with keys: hex_features, vertex_features, edge_features,
        scalars, legal_mask.
      - reward is +1 if the current player just won, -1 if game ended without
        them winning, 0 otherwise.
      - done is True iff the game is terminal.
      - info contains stats snapshot and current_player.
    """

    def __init__(self, seed: int = 0):
        self._engine = _Engine(seed)

    def reset(self, seed: int = 0):
        self._engine = _Engine(seed)
        return self._observation()

    def step(self, action_id: int):
        prev_player = self._engine.current_player()
        self._engine.step(int(action_id))
        done = self._engine.is_terminal()
        reward = 0.0
        if done:
            stats = self._engine.stats()
            winner = stats["winner_player_id"]
            reward = 1.0 if winner == prev_player else -1.0
        return self._observation(), reward, done, {"current_player": self._engine.current_player()}

    def legal_actions(self) -> np.ndarray:
        return self._engine.legal_actions()

    def legal_mask(self) -> np.ndarray:
        return self._observation()["legal_mask"].astype(bool)

    def stats(self):
        return self._engine.stats()

    def is_terminal(self) -> bool:
        return self._engine.is_terminal()

    # --- Chance-node API (pass-through to the Rust engine) -------------------
    def is_chance_pending(self) -> bool:
        return self._engine.is_chance_pending()

    def chance_outcomes(self):
        return self._engine.chance_outcomes()

    def apply_chance_outcome(self, value: int) -> None:
        self._engine.apply_chance_outcome(int(value))

    def action_history(self):
        return self._engine.action_history()

    def _observation(self):
        obs = self._engine.observation()
        return obs
