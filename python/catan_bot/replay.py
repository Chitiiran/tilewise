"""Versioned replay log: (seed, action sequence) → reconstructable game."""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List
from catan_bot.env import CatanEnv

REPLAY_SCHEMA_VERSION = 1


@dataclass
class Replay:
    schema_version: int
    seed: int
    actions: List[int]
    engine_version: str
    rules_tier: int = 1

    def save(self, path: str | Path):
        Path(path).write_text(json.dumps(asdict(self)))

    @classmethod
    def load(cls, path: str | Path) -> "Replay":
        data = json.loads(Path(path).read_text())
        if data["schema_version"] != REPLAY_SCHEMA_VERSION:
            raise ValueError(f"Unsupported replay schema {data['schema_version']}")
        return cls(**data)

    def reconstruct(self) -> CatanEnv:
        env = CatanEnv(seed=self.seed)
        env.reset(seed=self.seed)
        for a in self.actions:
            env.step(a)
        return env
