"""Self-play recorder. Writes moves.parquet, games.parquet, config.json per run.

Schema is documented in docs/superpowers/specs/2026-04-27-mcts-study-design.md §1.5.
SCHEMA_VERSION must be bumped explicitly on any field change; tests assert the value.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator
import uuid

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from . import ACTION_SPACE_SIZE


SCHEMA_VERSION = 1


@dataclass
class _MoveRow:
    seed: int
    move_index: int
    current_player: int
    legal_action_mask: list[bool]
    mcts_visit_counts: list[int]
    action_taken: int
    mcts_root_value: float
    schema_version: int = SCHEMA_VERSION


@dataclass
class _GameRow:
    seed: int
    winner: int
    final_vp: list[int]
    length_in_moves: int
    mcts_config_id: str
    schema_version: int = SCHEMA_VERSION


class _GameRecorder:
    def __init__(self, parent: "SelfPlayRecorder", seed: int) -> None:
        self._parent = parent
        self._seed = seed
        self._moves: list[_MoveRow] = []
        self._finalized = False

    def record_move(
        self,
        *,
        current_player: int,
        move_index: int,
        legal_action_mask: np.ndarray,
        mcts_visit_counts: np.ndarray,
        action_taken: int,
        mcts_root_value: float,
    ) -> None:
        assert legal_action_mask.shape == (ACTION_SPACE_SIZE,)
        assert mcts_visit_counts.shape == (ACTION_SPACE_SIZE,)
        assert legal_action_mask[action_taken], (
            f"action_taken={action_taken} not in legal_action_mask"
        )
        self._moves.append(_MoveRow(
            seed=self._seed,
            move_index=move_index,
            current_player=current_player,
            legal_action_mask=[bool(x) for x in legal_action_mask],
            mcts_visit_counts=[int(x) for x in mcts_visit_counts],
            action_taken=int(action_taken),
            mcts_root_value=float(mcts_root_value),
        ))

    def finalize(self, *, winner: int, final_vp: list[int], length_in_moves: int) -> None:
        self._parent._game_rows.append(_GameRow(
            seed=self._seed,
            winner=int(winner),
            final_vp=[int(x) for x in final_vp],
            length_in_moves=int(length_in_moves),
            mcts_config_id=self._parent._config_id,
        ))
        self._parent._move_rows.extend(self._moves)
        self._finalized = True


class SelfPlayRecorder:
    def __init__(self, out_dir: Path, config: dict[str, Any]) -> None:
        self._out_dir = Path(out_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._config_id = str(uuid.uuid4())
        full_config = {**config, "mcts_config_id": self._config_id, "schema_version": SCHEMA_VERSION}
        (self._out_dir / "config.json").write_text(json.dumps(full_config, indent=2))
        self._move_rows: list[_MoveRow] = []
        self._game_rows: list[_GameRow] = []

    @contextmanager
    def game(self, seed: int) -> Iterator[_GameRecorder]:
        rec = _GameRecorder(self, seed)
        try:
            yield rec
        finally:
            if not rec._finalized:
                # Game ended without finalize — record a -1 winner
                rec.finalize(winner=-1, final_vp=[0]*4, length_in_moves=len(rec._moves))

    def flush(self) -> None:
        """Write the unlabeled `moves.parquet` + `games.parquet` for whatever's
        in the buffer. No-op if buffers are empty (so post-checkpoint flushes
        don't overwrite shards with empty files)."""
        if not self._move_rows and not self._game_rows:
            return
        moves_table = pa.Table.from_pylist([row.__dict__ for row in self._move_rows])
        games_table = pa.Table.from_pylist([row.__dict__ for row in self._game_rows])
        pq.write_table(moves_table, self._out_dir / "moves.parquet")
        pq.write_table(games_table, self._out_dir / "games.parquet")

    def checkpoint(self, label: str) -> None:
        """v2 hardening: write `moves.<label>.parquet` + `games.<label>.parquet`
        for the currently-buffered rows, then clear the buffers.

        Lets experiments flush per sims-grid cell (or per c-value, or per policy)
        so partial data survives a kill mid-run. Notebooks glob `moves.*.parquet`
        on read.

        No-op if buffers are empty.
        """
        if not self._move_rows and not self._game_rows:
            return
        moves_table = pa.Table.from_pylist([row.__dict__ for row in self._move_rows])
        games_table = pa.Table.from_pylist([row.__dict__ for row in self._game_rows])
        pq.write_table(moves_table, self._out_dir / f"moves.{label}.parquet")
        pq.write_table(games_table, self._out_dir / f"games.{label}.parquet")
        self._move_rows.clear()
        self._game_rows.clear()


def visit_counts_from_root(root) -> np.ndarray:
    """Extract a length-ACTION_SPACE_SIZE int32 array of root-child visit counts from
    OpenSpiel's MCTS search-tree root node.

    OpenSpiel's `SearchNode` exposes `.children` as a list of `SearchNode`, each with
    `.action` (int) and `.explore_count` (int).
    """
    out = np.zeros(ACTION_SPACE_SIZE, dtype=np.int32)
    for child in root.children:
        out[int(child.action)] = int(child.explore_count)
    return out
