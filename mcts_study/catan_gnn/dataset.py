"""CatanReplayDataset: stream training tuples from existing parquet runs.

For each MCTS-decided move row in moves.parquet, we:
  1. Find its game's full action_history (from games.parquet, schema v2).
  2. Replay the engine forward through the prefix of action_history that ends
     at this MCTS move.
  3. Build features via state_to_pyg(engine.observation()).
  4. Build value target from games.winner (perspective-rotated).
  5. Build policy target from mcts_visit_counts (normalized to a distribution).
  6. Return (HeteroData, value [4], policy [206], legal_mask [206]).

Skips any row whose game has schema_version != 2 -- those games don't have
action_history and would require expensive replay-from-scratch.

Replay note: in the canonical experiment loop (common.play_one_game),
`move_index` is incremented ONLY for the recorded player. So to re-find the
i-th recorded move for a given seed, we count non-trivial decisions where
`engine.current_player() == row.current_player`. Other players' decisions are
applied verbatim from action_history but don't advance the move counter.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from catan_bot import _engine

from .state_to_pyg import state_to_pyg


_CHANCE_BIT = 0x8000_0000


class CatanReplayDataset(Dataset):
    def __init__(self, run_dirs: list[Path]) -> None:
        moves_frames = []
        games_frames = []
        for rd in run_dirs:
            rd = Path(rd)
            for mv in rd.glob("worker*/moves.*.parquet"):
                moves_frames.append(pq.read_table(mv).to_pandas())
            for gv in rd.glob("worker*/games.*.parquet"):
                games_frames.append(pq.read_table(gv).to_pandas())
            # Also accept top-level shards (serial mode, e.g. tests).
            for mv in rd.glob("moves.*.parquet"):
                moves_frames.append(pq.read_table(mv).to_pandas())
            for gv in rd.glob("games.*.parquet"):
                games_frames.append(pq.read_table(gv).to_pandas())
        if not moves_frames:
            raise RuntimeError(f"No moves.*.parquet shards found under {run_dirs}")
        import pandas as pd
        moves = pd.concat(moves_frames, ignore_index=True)
        games = pd.concat(games_frames, ignore_index=True)
        # Filter to v2 games only (need action_history).
        games = games[games["schema_version"] >= 2]
        self._winner_by_seed = {int(s): int(w) for s, w in zip(games["seed"], games["winner"])}
        self._history_by_seed = {
            int(s): list(h) for s, h in zip(games["seed"], games["action_history"])
        }
        self._index = moves[moves["seed"].isin(self._winner_by_seed.keys())].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i: int):
        row = self._index.iloc[i]
        seed = int(row["seed"])
        move_index = int(row["move_index"])
        recorded_player = int(row["current_player"])
        full_history = self._history_by_seed[seed]

        # Replay. action_history records every engine action (chance has the
        # 0x80000000 high bit set; regular actions don't). Walk it forward,
        # counting non-trivial decisions made by `recorded_player` (since the
        # recorder's `move_index` only increments for that player). Stop just
        # BEFORE applying the move_index-th such decision.
        engine = _engine.Engine(seed)
        recorded_decisions_seen = 0
        stopped = False
        for action_id in full_history:
            if engine.is_terminal():
                break
            a = int(action_id)
            if engine.is_chance_pending():
                # Chance entry: history stores it with the high bit set.
                engine.apply_chance_outcome(a & 0x7FFF_FFFF)
                continue
            # Decision step. Is this the move we're looking for?
            legal = engine.legal_actions()
            if len(legal) > 1 and int(engine.current_player()) == recorded_player:
                if recorded_decisions_seen == move_index:
                    stopped = True
                    break
                recorded_decisions_seen += 1
            engine.step(a)

        if not stopped:
            raise RuntimeError(
                f"Could not replay seed={seed} to move_index={move_index} "
                f"(player={recorded_player}); only saw "
                f"{recorded_decisions_seen} matching decisions."
            )

        obs = engine.observation()
        data = state_to_pyg(obs)

        # Value target: perspective-rotated.
        # value[i] = +1 if (current_player + i) % 4 == winner, else -1.
        # If winner == -1 (timeout/draw), value is all zeros.
        winner = self._winner_by_seed[seed]
        value = torch.zeros(4, dtype=torch.float32)
        if winner != -1:
            for offset in range(4):
                player = (recorded_player + offset) % 4
                value[offset] = 1.0 if player == winner else -1.0

        # Policy target: visit-count distribution (normalized).
        visits = np.array(row["mcts_visit_counts"], dtype=np.float32)
        s = float(visits.sum())
        if s > 0:
            policy = torch.from_numpy(visits / s)
        else:
            # Degenerate fallback: uniform over legal actions.
            mask_arr = np.array(row["legal_action_mask"], dtype=bool)
            policy = torch.zeros(206, dtype=torch.float32)
            n_legal = int(mask_arr.sum())
            if n_legal > 0:
                policy[mask_arr] = 1.0 / n_legal

        legal_mask = torch.from_numpy(np.array(row["legal_action_mask"], dtype=np.bool_))

        return data, value, policy, legal_mask
