"""CatanReplayDataset: stream training tuples from existing parquet runs.

For each MCTS-decided move row in moves.parquet, we:
  1. Find its game's full action_history (from games.parquet, schema v2).
  2. Replay the engine forward through the prefix of action_history that ends
     at this MCTS move.
  3. Build features via state_to_pyg(engine.observation()).
  4. Build value target from games.winner (perspective-rotated).
  5. Build policy target from mcts_visit_counts (normalized to a distribution).
  6. Return (HeteroData, value [4], policy [ACTION_SPACE_SIZE], legal_mask [ACTION_SPACE_SIZE]).

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
from catan_mcts import ACTION_SPACE_SIZE

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
        if not moves_frames and not games_frames:
            raise RuntimeError(f"No moves.*.parquet shards found under {run_dirs}")
        import pandas as pd
        moves = pd.concat(moves_frames, ignore_index=True) if moves_frames else pd.DataFrame()
        games = pd.concat(games_frames, ignore_index=True) if games_frames else pd.DataFrame()
        # Filter to v2 games only (need action_history).
        if not games.empty:
            games = games[games["schema_version"] >= 2]
        if games.empty or "action_history" not in games.columns:
            self._winner_by_seed = {}
            self._history_by_seed = {}
            self._index = moves.iloc[0:0].reset_index(drop=True) if not moves.empty else moves
            return
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
            policy = torch.zeros(ACTION_SPACE_SIZE, dtype=torch.float32)
            n_legal = int(mask_arr.sum())
            if n_legal > 0:
                policy[mask_arr] = 1.0 / n_legal

        legal_mask = torch.from_numpy(np.array(row["legal_action_mask"], dtype=np.bool_))

        return data, value, policy, legal_mask


class CachedDataset(Dataset):
    """In-RAM cache wrapper around a slow Dataset (typically CatanReplayDataset).

    Why: CatanReplayDataset's __getitem__ replays the engine from scratch per
    call, which is CPU-bound and dominates training wall-clock (GPU ends up
    waiting). Replaying once and keeping the result in RAM removes that
    bottleneck for subsequent epochs. With optional disk persistence
    (via cache_path), a one-time precompute survives across training sessions.

    Storage format: HeteroData reconstructed on load from raw tensor dicts,
    avoiding PyG version drift in pickled HeteroData objects. Each cached
    item is a dict of tensors (no Python objects).

    Memory footprint: ~10-12 KB per position (3 small float matrices + scalars
    + legal_mask + 4-vector value + ACTION_SPACE_SIZE-vector policy). 100k positions ~= 1 GB.
    Fits comfortably in 16 GB RAM for any v0 dataset size.
    """

    def __init__(
        self,
        source: Dataset | None,
        cache_path: Path | None = None,
        verbose: bool = True,
    ) -> None:
        self._items: list[dict] = []
        # Per-position seed list, mirroring the source's index. Used by
        # train._split_by_seed to do whole-game train/val splits even when the
        # source CatanReplayDataset has been wrapped in this cache.
        self.seeds: list[int] = []
        if cache_path is not None:
            cache_path = Path(cache_path)

        if cache_path is not None and cache_path.exists():
            if verbose:
                print(f"[CachedDataset] loading cache from {cache_path}", flush=True)
            blob = torch.load(cache_path, map_location="cpu", weights_only=False)
            self._items = blob["items"]
            self.seeds = list(blob.get("seeds", []))
            if verbose:
                print(f"[CachedDataset] loaded {len(self._items)} positions", flush=True)
            return

        if source is None:
            raise RuntimeError(
                f"CachedDataset: no cache at {cache_path} and no source dataset provided"
            )

        # Pull seed-per-position from the source if it has _index (CatanReplayDataset).
        source_seeds: list[int] | None = None
        if hasattr(source, "_index"):
            source_seeds = [int(s) for s in source._index["seed"]]

        n = len(source)
        if verbose:
            print(f"[CachedDataset] building cache from {n} source positions", flush=True)
        import time as _time
        t0 = _time.perf_counter()
        for i in range(n):
            data, value, policy, legal_mask = source[i]
            self._items.append({
                "hex_x": data["hex"].x.contiguous(),
                "vertex_x": data["vertex"].x.contiguous(),
                "edge_x": data["edge"].x.contiguous(),
                "scalars": data.scalars.contiguous(),
                "legal_mask_attr": data.legal_mask.contiguous(),
                "h2v_ei": data["hex", "to", "vertex"].edge_index,
                "v2h_ei": data["vertex", "to", "hex"].edge_index,
                "v2e_ei": data["vertex", "to", "edge"].edge_index,
                "e2v_ei": data["edge", "to", "vertex"].edge_index,
                "value": value,
                "policy": policy,
                "legal": legal_mask,
            })
            if source_seeds is not None:
                self.seeds.append(source_seeds[i])
            if verbose and (i + 1) % 500 == 0:
                rate = (i + 1) / (_time.perf_counter() - t0)
                eta_s = (n - (i + 1)) / max(rate, 1e-6)
                print(f"[CachedDataset] built {i + 1}/{n} "
                      f"({rate:.1f} pos/s, eta {eta_s:.0f}s)", flush=True)

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"items": self._items, "seeds": self.seeds, "version": 2},
                       cache_path)
            if verbose:
                print(f"[CachedDataset] saved cache to {cache_path}", flush=True)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, i: int):
        from torch_geometric.data import HeteroData
        it = self._items[i]
        data = HeteroData()
        data["hex"].x = it["hex_x"]
        data["vertex"].x = it["vertex_x"]
        data["edge"].x = it["edge_x"]
        data["hex", "to", "vertex"].edge_index = it["h2v_ei"]
        data["vertex", "to", "hex"].edge_index = it["v2h_ei"]
        data["vertex", "to", "edge"].edge_index = it["v2e_ei"]
        data["edge", "to", "vertex"].edge_index = it["e2v_ei"]
        data.scalars = it["scalars"]
        data.legal_mask = it["legal_mask_attr"]
        return data, it["value"], it["policy"], it["legal"]


class RotatedDataset(Dataset):
    """Wraps a CachedDataset and applies a hex rotation to every item.

    Same length as the source. Each `__getitem__` rotates the HeteroData
    (hex/vertex/edge feature rows + edge_index entries), the policy target,
    and the legal mask. Value targets are unchanged (rotation doesn't affect
    who won).

    Modes:
      - mode="fixed", k=N (default 1): always apply k×60° rotation.
        k=0 is identity; k=1..5 are the five non-trivial hex symmetries.
      - mode="random": each __getitem__ picks k uniformly from 0..5. The full
        6-fold hex symmetry group, including identity. Standard data
        augmentation pattern — every batch sees a mix of orientations.
    """

    def __init__(self, source: Dataset, mode: str = "fixed", k: int = 1,
                 seed: int | None = None) -> None:
        self.source = source
        if mode not in ("fixed", "random"):
            raise ValueError(f"mode must be 'fixed' or 'random', got {mode!r}")
        self.mode = mode
        self.k = int(k) % 6
        # Per-instance RNG so DataLoader workers don't all roll the same k.
        # If seed is None, use Python's default (different per worker thanks
        # to torch's worker_init_fn or numpy's random_state).
        import random as _random
        self._rng = _random.Random(seed)
        if hasattr(source, "seeds"):
            self.seeds = source.seeds

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, i: int):
        from .rotation import (
            rotate_hetero_data_k,
            rotate_legal_mask_k,
            rotate_policy_k,
        )

        if self.mode == "random":
            k = self._rng.randrange(6)
        else:
            k = self.k

        data, value, policy, legal = self.source[i]
        if k == 0:
            return data, value, policy, legal
        rotated = rotate_hetero_data_k(data, k)
        return rotated, value, rotate_policy_k(policy, k), rotate_legal_mask_k(legal, k)
