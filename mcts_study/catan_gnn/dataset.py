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

import logging
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
        # Dedup guard: if multiple self-play procs collide into one run dir
        # (make_run_dir minute-resolution, hit 2026-06-11), their racing
        # end-of-run consolidations duplicate rows exactly. A duplicated
        # position would get double training weight. Each (seed, move_index,
        # current_player) is one decision, so dedup on that key is lossless.
        # (games dups already collapse via the seed-keyed dicts below.)
        if not moves.empty:
            moves = moves.drop_duplicates(
                subset=["seed", "move_index", "current_player"]
            ).reset_index(drop=True)
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
        # Build-time divergence filter (2026-06-14): for some self-play games the
        # action_history replay reproduces FEWER gated decisions for a player
        # than the recorder logged move rows (recorder<->replay determinism
        # divergence). Those move rows can never be replayed and would crash
        # training. Replay each game ONCE (per-game, not per-position) to get
        # the replayable decision count per player, then drop out-of-range rows.
        # ~0.05% of positions dropped on the iter-3 corpus. See
        # 2026-06-14-recorder-replay-divergence-and-recording-roadmap.md.
        counts_by_seed = {
            seed: replayable_decision_counts(seed=seed, history=hist)
            for seed, hist in self._history_by_seed.items()
        }
        before = len(self._index)
        self._index = _drop_divergent_rows(self._index, counts_by_seed
                                           ).reset_index(drop=True)
        dropped = before - len(self._index)
        if dropped:
            logging.warning("dataset: dropped %d/%d divergent (un-replayable) "
                            "positions (%.3f%%) — recorder<->replay divergence",
                            dropped, before, 100.0 * dropped / max(1, before))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i: int):
        # Resilience (2026-06-14): a single un-replayable position must not
        # crash training — substitute the next valid neighbor + log. A rare,
        # non-reproducible replay failure on one self-play game killed iter-3
        # training after 6h of self-play; defense-in-depth (spec 2026-06-13 §2)
        # turns that fatal error into a skipped sample.
        return resilient_getitem(self, i)

    def _replay_getitem(self, i: int):
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


def replayable_decision_counts(*, seed: int, history) -> dict:
    """Replay a game's action_history once and return, per player, how many
    GATED decisions (len(legal) > 1, that player to move) the history actually
    reproduces. This is the recorder's move_index ceiling per player: a move
    row with move_index >= this count diverges and can't be replayed.

    Mirrors CatanReplayDataset's replay gating EXACTLY so the counts match what
    __getitem__ would reach. (2026-06-14 divergence filter.)"""
    engine = _engine.Engine(int(seed))
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for action_id in history:
        if engine.is_terminal():
            break
        a = int(action_id)
        if engine.is_chance_pending():
            engine.apply_chance_outcome(a & 0x7FFF_FFFF)
            continue
        legal = engine.legal_actions()
        if len(legal) > 1:
            counts[int(engine.current_player())] += 1
        engine.step(a)
    return counts


def _drop_divergent_rows(moves, counts_by_seed):
    """Drop move rows whose move_index >= the replayable decision count for
    that (seed, current_player). Vectorized over the moves frame."""
    import numpy as np
    ceil = np.array([
        counts_by_seed.get(int(s), {}).get(int(p), 0)
        for s, p in zip(moves["seed"], moves["current_player"])
    ])
    keep = moves["move_index"].to_numpy() < ceil
    return moves[keep]


def resilient_getitem(ds, i: int, max_tries: int = 64):
    """Return ds._replay_getitem(i), or the next valid neighbor if it can't be
    replayed. A single un-replayable position (rare, non-reproducible data/
    engine glitch — 2026-06-14 iter-3) must NOT crash training; we skip-and-
    substitute + log, keeping the batch full. Raises only if `max_tries`
    consecutive positions all fail (genuine corruption, not a one-off)."""
    n = len(ds)
    for k in range(max_tries):
        j = (i + k) % n
        try:
            return ds._replay_getitem(j)
        except Exception as ex:  # noqa: BLE001 — any replay failure is skippable
            if k == 0:
                logging.warning("dataset: skipping un-replayable position %d "
                                "(%s); substituting a neighbor", i, ex)
            continue
    raise RuntimeError(
        f"dataset: no replayable position within {max_tries} of index {i} — "
        f"this indicates real corruption, not a one-off skip.")


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

    Sparse format (sparse=True, default for new caches): the 4 edge_index
    tensors (hex<->vertex, vertex<->edge) are NOT stored per sample because
    Catan board geometry is constant. They live as a class-level singleton
    `_SHARED_EDGE_INDEX` and are referenced (not copied) by every __getitem__
    call. Saves ~30 GB of redundant storage on a 100k corpus (3.2M samples).

    Memory footprint:
      - dense (legacy): ~16 KB per sample. 3.2M positions ~= 51 GB.
      - sparse (new): ~7 KB per sample. 3.2M positions ~= 22 GB.
    """

    # Class-level singleton: the fixed Catan-board edge_index tables.
    # Lazily computed on first instantiation. Shared across all instances
    # and all samples (read-only — RotatedDataset never mutates these,
    # it builds new edge_index tensors via torch.stack).
    _SHARED_EDGE_INDEX: dict | None = None

    @classmethod
    def _get_shared_edge_index(cls) -> dict:
        """Compute (or return cached) the 4 edge_index tables for an empty
        Catan board. Identical for every game position."""
        if cls._SHARED_EDGE_INDEX is not None:
            return cls._SHARED_EDGE_INDEX
        from .state_to_pyg import state_to_pyg
        eng = _engine.Engine(0)
        ref_data = state_to_pyg(eng.observation())
        cls._SHARED_EDGE_INDEX = {
            "h2v_ei": ref_data["hex", "to", "vertex"].edge_index.clone(),
            "v2h_ei": ref_data["vertex", "to", "hex"].edge_index.clone(),
            "v2e_ei": ref_data["vertex", "to", "edge"].edge_index.clone(),
            "e2v_ei": ref_data["edge", "to", "vertex"].edge_index.clone(),
        }
        return cls._SHARED_EDGE_INDEX

    def __init__(
        self,
        source: Dataset | None,
        cache_path: Path | None = None,
        verbose: bool = True,
        chunk_size: int = 500_000,
        sparse: bool = True,
        eager_load: bool = True,
    ) -> None:
        self._items: list[dict] = []
        # When True, items don't carry h2v_ei/v2h_ei/v2e_ei/e2v_ei; __getitem__
        # reads them from the class-level _SHARED_EDGE_INDEX. Set during cache
        # build; preserved across save/load via the manifest's "format" field.
        self._sparse: bool = sparse
        # Per-position seed list, mirroring the source's index. Used by
        # train._split_by_seed to do whole-game train/val splits even when the
        # source CatanReplayDataset has been wrapped in this cache.
        self.seeds: list[int] = []
        if cache_path is not None:
            cache_path = Path(cache_path)

        if cache_path is not None and cache_path.exists():
            self._load_from_disk(cache_path, verbose=verbose)
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
            fmt = "sparse" if sparse else "dense"
            print(f"[CachedDataset] building {fmt} cache from {n} source positions"
                  f" (chunk_size={chunk_size})", flush=True)
        import time as _time
        t0 = _time.perf_counter()

        # Sparse mode: ensure the shared edge_index singleton exists, and
        # validate the first sample matches it (catches the unlikely case
        # where state_to_pyg produces something different).
        shared = self._get_shared_edge_index() if sparse else None

        use_chunks = cache_path is not None and n > chunk_size
        chunk_buf: list[dict] = []
        chunk_seeds: list[int] = []
        chunk_paths: list[tuple[str, int]] = []  # (path, length)

        for i in range(n):
            data, value, policy, legal_mask = source[i]
            item = {
                "hex_x": data["hex"].x.contiguous(),
                "vertex_x": data["vertex"].x.contiguous(),
                "edge_x": data["edge"].x.contiguous(),
                "scalars": data.scalars.contiguous(),
                "legal_mask_attr": data.legal_mask.contiguous(),
                "value": value,
                "policy": policy,
                "legal": legal_mask,
            }
            if not sparse:
                # Dense path: store per-sample edge_index too
                item["h2v_ei"] = data["hex", "to", "vertex"].edge_index
                item["v2h_ei"] = data["vertex", "to", "hex"].edge_index
                item["v2e_ei"] = data["vertex", "to", "edge"].edge_index
                item["e2v_ei"] = data["edge", "to", "vertex"].edge_index
            elif i == 0:
                # Sparse-mode invariant check: first sample's edge_index
                # must match shared. Fail loud if not.
                src_h2v = data["hex", "to", "vertex"].edge_index
                if not torch.equal(src_h2v, shared["h2v_ei"]):
                    raise RuntimeError(
                        "CachedDataset(sparse=True): first sample's h2v_ei "
                        "differs from the shared global edge_index. Cannot "
                        "use sparse cache."
                    )
            seed = source_seeds[i] if source_seeds is not None else None

            if use_chunks:
                chunk_buf.append(item)
                if seed is not None:
                    chunk_seeds.append(seed)
                if len(chunk_buf) >= chunk_size:
                    chunk_paths.append(self._flush_chunk(
                        cache_path, chunk_buf, chunk_seeds, len(chunk_paths), verbose
                    ))
                    chunk_buf = []
                    chunk_seeds = []
            else:
                self._items.append(item)
                if seed is not None:
                    self.seeds.append(seed)

            if verbose and (i + 1) % 500 == 0:
                rate = (i + 1) / (_time.perf_counter() - t0)
                eta_s = (n - (i + 1)) / max(rate, 1e-6)
                print(f"[CachedDataset] built {i + 1}/{n} "
                      f"({rate:.1f} pos/s, eta {eta_s:.0f}s)", flush=True)

        if use_chunks:
            # Flush any final partial chunk
            if chunk_buf:
                chunk_paths.append(self._flush_chunk(
                    cache_path, chunk_buf, chunk_seeds, len(chunk_paths), verbose
                ))
                chunk_buf = []
                chunk_seeds = []
            # Write manifest at cache_path itself.
            # Format string encodes both chunking and storage layout:
            #   chunked-v1            — legacy dense, per-sample edge_index
            #   chunked-v2-sparse     — new sparse, edge_index shared globally
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            total = sum(n for _, n in chunk_paths)
            fmt_str = "chunked-v2-sparse" if sparse else "chunked-v1"
            torch.save(
                {"format": fmt_str, "chunks": chunk_paths, "total": total,
                 "sparse": bool(sparse)},
                cache_path,
            )
            if verbose:
                print(f"[CachedDataset] wrote {fmt_str} manifest to {cache_path} "
                      f"({len(chunk_paths)} chunks, {total} items)", flush=True)
            # Eager-load chunks back into one list so __getitem__ works in the
            # same process. Skip when eager_load=False (build-and-exit flow:
            # the build script doesn't need a usable dataset, just files on
            # disk). Skipping avoids a memory spike that has crashed WSL.
            if eager_load:
                self._load_chunks(chunk_paths, verbose=verbose)
        elif cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            # Monolithic save also tags sparse vs dense.
            torch.save({"items": self._items, "seeds": self.seeds,
                        "version": 2, "sparse": bool(sparse)},
                       cache_path)
            if verbose:
                fmt = "sparse" if sparse else "dense"
                print(f"[CachedDataset] saved {fmt} cache to {cache_path}", flush=True)

    @staticmethod
    def _flush_chunk(cache_path: Path, items: list[dict], seeds: list[int],
                     chunk_idx: int, verbose: bool) -> tuple[str, int]:
        """Save one chunk to disk and return (path, length). Caller drops the
        in-memory buffer after this returns."""
        chunk_path = cache_path.parent / f"{cache_path.stem}_part{chunk_idx:03d}.pt"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"items": items, "seeds": seeds, "version": 2}, chunk_path)
        if verbose:
            print(f"[CachedDataset] flushed chunk {chunk_idx} -> {chunk_path.name} "
                  f"({len(items)} items, {chunk_path.stat().st_size / 1024**3:.2f} GB)",
                  flush=True)
        return (str(chunk_path), len(items))

    def _load_chunks(self, chunk_paths: list[tuple[str, int]], verbose: bool) -> None:
        """Eagerly load all chunks into self._items / self.seeds."""
        import time as _time
        t0 = _time.perf_counter()
        for idx, (path, n) in enumerate(chunk_paths):
            blob = torch.load(path, map_location="cpu", weights_only=False)
            self._items.extend(blob["items"])
            self.seeds.extend(blob.get("seeds", []))
            if verbose:
                print(f"[CachedDataset] loaded chunk {idx + 1}/{len(chunk_paths)} "
                      f"({n} items, total {len(self._items)})", flush=True)
        if verbose:
            print(f"[CachedDataset] all chunks loaded in {_time.perf_counter() - t0:.1f}s",
                  flush=True)

    def _load_from_disk(self, cache_path: Path, verbose: bool) -> None:
        """Load either a chunked manifest or a monolithic cache file.

        Detects format from manifest's "format"/"sparse" fields:
          - "chunked-v1"           → dense chunked
          - "chunked-v2-sparse"    → sparse chunked
          - monolithic w/ sparse=True → sparse monolithic
          - else                   → dense monolithic (legacy)
        """
        if verbose:
            print(f"[CachedDataset] loading cache from {cache_path}", flush=True)
        blob = torch.load(cache_path, map_location="cpu", weights_only=False)
        fmt = blob.get("format")
        # Recover sparse flag from manifest. Default False (dense) for
        # backward compat with old caches that have no flag.
        self._sparse = bool(blob.get("sparse", fmt == "chunked-v2-sparse"))
        if fmt in ("chunked-v1", "chunked-v2-sparse"):
            chunk_paths = blob["chunks"]
            if verbose:
                fmt_label = "sparse" if self._sparse else "dense"
                print(f"[CachedDataset] {fmt_label} chunked manifest found "
                      f"({len(chunk_paths)} chunks, {blob['total']} items)", flush=True)
            self._load_chunks(chunk_paths, verbose=verbose)
        else:
            self._items = blob["items"]
            self.seeds = list(blob.get("seeds", []))
            if verbose:
                fmt_label = "sparse" if self._sparse else "dense"
                print(f"[CachedDataset] loaded {len(self._items)} positions ({fmt_label})", flush=True)
        # If we're sparse, ensure the shared edge_index is materialized so
        # __getitem__ can use it. Cheap (one engine call).
        if self._sparse:
            self._get_shared_edge_index()

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, i: int):
        from torch_geometric.data import HeteroData
        it = self._items[i]
        data = HeteroData()
        data["hex"].x = it["hex_x"]
        data["vertex"].x = it["vertex_x"]
        data["edge"].x = it["edge_x"]
        # Sparse items lack edge_index keys; use the class-level shared store.
        # Per-item dense path stays bit-identical to the legacy behavior.
        if self._sparse:
            shared = self._SHARED_EDGE_INDEX
            data["hex", "to", "vertex"].edge_index = shared["h2v_ei"]
            data["vertex", "to", "hex"].edge_index = shared["v2h_ei"]
            data["vertex", "to", "edge"].edge_index = shared["v2e_ei"]
            data["edge", "to", "vertex"].edge_index = shared["e2v_ei"]
        else:
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
