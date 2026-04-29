"""v2 multiprocessing helper for compute-bound MCTS experiments.

The MCTS process is single-threaded (OpenSpiel's MCTSBot doesn't parallelize
simulations). But games are independent — each `run_one_game(seed, config)`
worker can run on its own core.

Pattern:
  - Top-level worker `_play_one_seed_for_pool` is pickle-safe.
  - Each worker creates its own `SelfPlayRecorder(out_dir / f"worker{n}", ...)`
    so parquet shards never collide.
  - Notebook globs `moves.worker*.parquet` to read all shards.

This module deliberately doesn't import `os_mcts.MCTSBot` at module top —
keeps imports lazy so subprocess startup is faster.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Callable

# A worker-result is whatever the experiment-specific function returns. We
# don't define a tight type here; experiment scripts cast.
WorkerResult = dict[str, Any]


@dataclass
class ParallelConfig:
    """Knobs for the pool runner."""
    out_dir: Path
    n_workers: int
    cell_label: str  # e.g. "sims=5" — appended to shard filenames
    config: dict[str, Any]  # passed through to each worker

    def worker_out_dir(self, worker_idx: int) -> Path:
        d = self.out_dir / f"worker{worker_idx}"
        d.mkdir(parents=True, exist_ok=True)
        return d


def run_parallel(
    seeds: list[int],
    worker_fn: Callable[..., WorkerResult],
    pcfg: ParallelConfig,
) -> list[WorkerResult]:
    """Run `worker_fn(seed, pcfg)` across `pcfg.n_workers` processes.

    Each worker process is given its own slice of seeds. Workers write to
    their own subdirectory (`out_dir/workerN/`) and return a list of
    summary dicts back to the parent.

    `worker_fn` MUST be a top-level (pickle-safe) function — defined at
    module scope, no closures.
    """
    if pcfg.n_workers <= 0:
        raise ValueError("n_workers must be >= 1")

    pcfg.out_dir.mkdir(parents=True, exist_ok=True)

    if pcfg.n_workers == 1:
        # Avoid pool overhead for serial runs (also helps debugging).
        results = [worker_fn(seed, pcfg, 0) for seed in seeds]
        return results

    # Slice seeds into n_workers contiguous batches; each worker handles
    # its own slice end-to-end.
    n = pcfg.n_workers
    chunks: list[list[int]] = [[] for _ in range(n)]
    for i, s in enumerate(seeds):
        chunks[i % n].append(s)

    ctx = get_context("spawn")  # cleanest cross-platform; needed on Windows + WSL
    args_list = [(chunk, worker_fn, pcfg, idx) for idx, chunk in enumerate(chunks)]
    with ctx.Pool(processes=n) as pool:
        per_worker = pool.map(_run_chunk, args_list)
    # Flatten per-worker result lists into one list, preserving worker order.
    return [r for sub in per_worker for r in sub]


def _run_chunk(args) -> list[WorkerResult]:
    """Pool-mapped helper: each call runs one chunk of seeds in one worker."""
    seeds, worker_fn, pcfg, worker_idx = args
    return [worker_fn(seed, pcfg, worker_idx) for seed in seeds]


# ---------- Reference worker functions (used by tests + experiment scripts) -


def random_game_worker(seed: int, pcfg: ParallelConfig, worker_idx: int) -> WorkerResult:
    """Tiny reference worker: plays one all-random game, returns the outcome.
    Used by tests; experiment scripts implement their own (with MCTS)."""
    import random
    from catan_mcts.adapter import CatanGame
    from .common import play_one_game

    class _RandomBot:
        def __init__(self, s):
            self._rng = random.Random(s)
        def step(self, state):
            return self._rng.choice(state.legal_actions())

    game = CatanGame()
    bots = {i: _RandomBot(seed + i) for i in range(4)}
    outcome = play_one_game(
        game=game, bots=bots, seed=seed, chance_rng=random.Random(seed),
        recorded_player=None, recorder_game=None,
        max_seconds=pcfg.config.get("max_seconds"),
    )
    return {
        "seed": seed,
        "winner": outcome.winner,
        "length_in_moves": outcome.length_in_moves,
        "timed_out": outcome.timed_out,
        "worker_idx": worker_idx,
        "pid": os.getpid(),
    }
