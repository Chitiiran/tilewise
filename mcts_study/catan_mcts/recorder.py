"""Self-play recorder. Writes per-game parquet shards for crash safety.

Schema documented in docs/superpowers/specs/2026-04-27-mcts-study-design.md §1.5.
SCHEMA_VERSION must be bumped explicitly on any field change; tests assert the value.

## Crash safety / data salvage (added 2026-05-01)

The recorder writes a per-seed parquet shard (`games.seed=<N>.parquet` +
`moves.seed=<N>.parquet`) the moment a game finalizes (or is skipped).
This means:

  * SIGINT / OOM / crash mid-sweep loses at most ONE in-flight game,
    not the whole cell's buffer.
  * `checkpoint(label)` and `flush()` keep their original API but become
    *compaction* operations: they read all per-seed shards, write a single
    labeled shard, then remove the per-seed files.
  * `skip_game()` now persists the partial action_history + moves rows
    the engine already produced, with `timed_out=True`. Long timed-out
    games are often the strategically-interesting ones (resource starvation,
    road blocking) — losing them was the v1 anti-pattern this fixes.
"""
from __future__ import annotations

import atexit
import json
import signal
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional
import uuid

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from . import ACTION_SPACE_SIZE


SCHEMA_VERSION = 2


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
    action_history: list[int]
    timed_out: bool = False
    schema_version: int = SCHEMA_VERSION


def _shard_paths(out_dir: Path, seed: int) -> tuple[Path, Path]:
    """Per-seed shard file paths."""
    return (
        out_dir / f"games.seed={seed}.parquet",
        out_dir / f"moves.seed={seed}.parquet",
    )


def _write_shard(out_dir: Path, seed: int, game_row: _GameRow,
                 move_rows: list[_MoveRow]) -> None:
    """Atomically write per-seed games + moves shards.

    Uses .tmp + os.replace pattern so a crash mid-write leaves the previous
    shard intact (or no shard at all), never a half-written one.
    """
    games_path, moves_path = _shard_paths(out_dir, seed)
    games_table = pa.Table.from_pylist([game_row.__dict__])
    moves_table = pa.Table.from_pylist([row.__dict__ for row in move_rows])
    games_tmp = games_path.with_suffix(".tmp.parquet")
    moves_tmp = moves_path.with_suffix(".tmp.parquet")
    pq.write_table(games_table, games_tmp)
    pq.write_table(moves_table, moves_tmp)
    games_tmp.replace(games_path)
    moves_tmp.replace(moves_path)


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

    def finalize(self, *, winner: int, final_vp: list[int], length_in_moves: int,
                 action_history: list[int], timed_out: bool = False) -> None:
        """Write the per-seed parquet shard immediately. After this call the
        game's data is on disk; no buffered rows can be lost to a crash."""
        game_row = _GameRow(
            seed=self._seed,
            winner=int(winner),
            final_vp=[int(x) for x in final_vp],
            length_in_moves=int(length_in_moves),
            mcts_config_id=self._parent._config_id,
            action_history=[int(x) for x in action_history],
            timed_out=bool(timed_out),
        )
        _write_shard(self._parent._out_dir, self._seed, game_row, self._moves)
        self._finalized = True


class SelfPlayRecorder:
    def __init__(self, out_dir: Path, config: dict[str, Any]) -> None:
        self._out_dir = Path(out_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._config_id = str(uuid.uuid4())
        full_config = {**config, "mcts_config_id": self._config_id, "schema_version": SCHEMA_VERSION}
        (self._out_dir / "config.json").write_text(json.dumps(full_config, indent=2))
        # Atexit hook to compact remaining per-seed shards into a single
        # `_remainder.parquet` if the process exits without an explicit
        # checkpoint() / flush(). This is defense-in-depth — the per-seed
        # shards are already on disk, so this is purely a convenience.
        atexit.register(self._atexit_compact)
        self._sigint_installed = False
        self._install_sigint_handler()

    def _install_sigint_handler(self) -> None:
        """Install a SIGINT handler that re-raises KeyboardInterrupt cleanly
        AFTER giving the per-seed shards a chance to finish flushing.
        Idempotent — calling __init__ multiple times in tests is safe."""
        if self._sigint_installed:
            return
        prev_handler = signal.getsignal(signal.SIGINT)
        def _handler(signum, frame):
            # Per-seed shards are already on disk; this just ensures the
            # in-flight game's finalize is given a final chance via the
            # normal exception path.
            if callable(prev_handler):
                prev_handler(signum, frame)
            else:
                raise KeyboardInterrupt
        try:
            signal.signal(signal.SIGINT, _handler)
            self._sigint_installed = True
        except (ValueError, OSError):
            # SIGINT can't be installed in non-main threads (pytest workers,
            # spawned mp processes, jupyter kernels). Per-seed flushing is
            # the primary safety net; SIGINT install is bonus.
            pass

    @contextmanager
    def game(self, seed: int) -> Iterator[_GameRecorder]:
        rec = _GameRecorder(self, seed)
        try:
            yield rec
        finally:
            if not rec._finalized:
                # Game ended without finalize — record a -1 winner with whatever
                # rows we have. Caller may also have called skip_game() which
                # already wrote the shard; skip if so.
                games_path, _ = _shard_paths(self._out_dir, seed)
                if not games_path.exists():
                    rec.finalize(
                        winner=-1, final_vp=[0]*4,
                        length_in_moves=len(rec._moves), action_history=[],
                    )

    def skip_game(self, *, seed: int, reason: str, length_in_moves: int = 0,
                  action_history: Optional[list[int]] = None,
                  moves_recorder: Optional[_GameRecorder] = None) -> None:
        """Record an abandoned/timed-out game.

        Always appends a row to skipped.csv (the human-readable index of
        non-completing games). When `action_history` is supplied, ALSO writes
        a per-seed parquet shard with `timed_out=True` so the game's data
        stays inspectable. `moves_recorder` (typically the active
        _GameRecorder) lets us preserve the partial moves rows too.

        Long timed-out games are often the strategically-interesting ones
        (resource starvation, road-blocking) — keeping their data is the
        whole point of this v2 hardening pass.
        """
        # Skipped.csv entry (always written — the human-readable index).
        csv_path = self._out_dir / "skipped.csv"
        is_new = not csv_path.exists()
        with csv_path.open("a", encoding="utf-8") as f:
            if is_new:
                f.write("seed,reason,length_in_moves\n")
            f.write(f"{int(seed)},{reason},{int(length_in_moves)}\n")

        # Per-seed parquet shard (only when caller hands us the data).
        if action_history is not None:
            move_rows = list(moves_recorder._moves) if moves_recorder is not None else []
            game_row = _GameRow(
                seed=int(seed),
                winner=-1,
                final_vp=[0, 0, 0, 0],
                length_in_moves=int(length_in_moves),
                mcts_config_id=self._config_id,
                action_history=[int(x) for x in action_history],
                timed_out=True,
            )
            _write_shard(self._out_dir, int(seed), game_row, move_rows)
            if moves_recorder is not None:
                moves_recorder._finalized = True

    def done_seeds(self) -> set[int]:
        """Read seeds that finished cleanly in this run dir. Used by experiments
        to skip already-completed work on restart."""
        done_path = self._out_dir / "done.txt"
        if not done_path.exists():
            return set()
        return {
            int(line.strip())
            for line in done_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

    def mark_done(self, seed: int) -> None:
        """Append `seed` to done.txt. Idempotent (downstream done_seeds()
        returns a set, so duplicate appends cause no harm)."""
        with (self._out_dir / "done.txt").open("a", encoding="utf-8") as f:
            f.write(f"{int(seed)}\n")

    def _read_per_seed_shards(self) -> tuple[list[pa.Table], list[pa.Table], list[int]]:
        """Read all per-seed shards in this dir. Returns (games_tables,
        moves_tables, seeds) sorted by seed."""
        game_files = sorted(self._out_dir.glob("games.seed=*.parquet"))
        move_files = sorted(self._out_dir.glob("moves.seed=*.parquet"))
        # Match by filename stem; we trust the `_write_shard` invariant that
        # both files always exist together.
        seeds = []
        games_tables = []
        moves_tables = []
        for gf in game_files:
            seed_str = gf.stem.removeprefix("games.seed=")
            try:
                seed = int(seed_str)
            except ValueError:
                continue
            mf = self._out_dir / f"moves.seed={seed}.parquet"
            if not mf.exists():
                continue
            seeds.append(seed)
            games_tables.append(pq.read_table(gf))
            moves_tables.append(pq.read_table(mf))
        return games_tables, moves_tables, seeds

    def _delete_per_seed_shards(self, seeds: list[int]) -> None:
        for seed in seeds:
            gf, mf = _shard_paths(self._out_dir, seed)
            try:
                gf.unlink(missing_ok=True)
                mf.unlink(missing_ok=True)
            except OSError:
                # Best-effort cleanup; combined shard is the source of truth.
                pass

    def _compact_to(self, label: str) -> bool:
        """Read all per-seed shards, concat, write a single labeled shard,
        delete the per-seed files. Returns True if any data was written.

        `label` is e.g. "sims=5" → writes games.sims=5.parquet + moves.sims=5.parquet,
                       "" (empty) → writes games.parquet + moves.parquet.
        """
        games_tables, moves_tables, seeds = self._read_per_seed_shards()
        if not games_tables:
            return False
        games_combined = pa.concat_tables(games_tables, promote_options="default")
        moves_combined = pa.concat_tables(moves_tables, promote_options="default") \
            if moves_tables else pa.Table.from_pylist([], schema=games_combined.schema.empty_table().schema if False else None)
        # Build moves table even if all rows are empty
        if not moves_tables:
            moves_combined = pa.table({})
        suffix = f".{label}" if label else ""
        games_path = self._out_dir / f"games{suffix}.parquet"
        moves_path = self._out_dir / f"moves{suffix}.parquet"
        pq.write_table(games_combined, games_path)
        # Some moves shards may be empty (no MCTS rows) — concat handles it.
        # Only write if we have at least an empty schema match.
        if moves_tables:
            pq.write_table(moves_combined, moves_path)
        else:
            # Write an empty moves file with the canonical schema by
            # reading any moves shard's schema. Fall back to single-row
            # template if no shard exists.
            empty_template = pa.table({
                "seed": pa.array([], type=pa.int64()),
                "schema_version": pa.array([], type=pa.int64()),
            })
            pq.write_table(empty_template, moves_path)
        self._delete_per_seed_shards(seeds)
        return True

    def checkpoint(self, label: str) -> None:
        """Compact all per-seed shards into `games.<label>.parquet` +
        `moves.<label>.parquet`, then delete the per-seed files.

        No-op if no per-seed shards exist.
        """
        self._compact_to(label)

    def flush(self) -> None:
        """Compact remaining per-seed shards into the unlabeled
        `games.parquet` / `moves.parquet`. No-op if none remain."""
        self._compact_to("")

    def _atexit_compact(self) -> None:
        """Atexit fallback: if any per-seed shards remain (because the user
        forgot to call flush() or the process crashed), compact them to a
        `_remainder.parquet` shard so they're discoverable. Best-effort —
        any exception here is suppressed (we're already exiting)."""
        try:
            self._compact_to("_remainder")
        except Exception:
            pass


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
