"""Tests for the v2 multiprocessing pool runner."""
from __future__ import annotations

from pathlib import Path

from catan_mcts.experiments.parallel import (
    ParallelConfig, run_parallel, random_game_worker,
)


def test_parallel_serial_mode_runs_each_seed_once(tmp_path: Path):
    """n_workers=1 takes the bypass path (no Pool); each seed handled exactly once."""
    pcfg = ParallelConfig(
        out_dir=tmp_path, n_workers=1, cell_label="smoke",
        config={"max_seconds": 2.0},
    )
    seeds = [1, 2, 3]
    results = run_parallel(seeds, random_game_worker, pcfg)
    assert len(results) == 3
    seeds_back = sorted(int(r["seed"]) for r in results)
    assert seeds_back == [1, 2, 3]


def test_parallel_pool_mode_distributes_across_workers(tmp_path: Path):
    """n_workers=2 actually spawns subprocesses with distinct PIDs."""
    pcfg = ParallelConfig(
        out_dir=tmp_path, n_workers=2, cell_label="smoke",
        config={"max_seconds": 2.0},
    )
    seeds = [10, 20, 30, 40]
    results = run_parallel(seeds, random_game_worker, pcfg)
    assert len(results) == 4
    seeds_back = sorted(int(r["seed"]) for r in results)
    assert seeds_back == [10, 20, 30, 40]
    # At least 2 distinct PIDs and 2 distinct worker_idx values.
    pids = {r["pid"] for r in results}
    workers = {r["worker_idx"] for r in results}
    assert len(pids) >= 2, f"expected pool to spawn workers but got pids={pids}"
    assert workers == {0, 1}


def test_parallel_max_seconds_terminates_workers_promptly(tmp_path: Path):
    """A tiny max_seconds in the config must propagate to workers and bound
    wall-clock. We can't insist *all* games time out (very short random games
    can finish in <50ms by chance), but we can insist the run finishes quickly."""
    import time
    pcfg = ParallelConfig(
        out_dir=tmp_path, n_workers=2, cell_label="smoke",
        config={"max_seconds": 0.05},  # 50 ms cap per game
    )
    seeds = [1, 2, 3, 4]
    t0 = time.perf_counter()
    results = run_parallel(seeds, random_game_worker, pcfg)
    elapsed = time.perf_counter() - t0
    # 4 games at 50ms cap × 2 workers should be well under 5 s including
    # subprocess spawn overhead. The bound just needs to demonstrate that
    # max_seconds is being applied — not absolute timing.
    assert elapsed < 30.0, f"parallel run with 50ms cap took {elapsed:.1f}s"
    assert sorted(int(r["seed"]) for r in results) == [1, 2, 3, 4]
    # Most games should have timed out (caught the cap rather than finished).
    timed_out = [r for r in results if r["timed_out"]]
    assert len(timed_out) >= 2, (
        f"expected most 4-game-x-50ms runs to time out, got {len(timed_out)}/4"
    )


def test_parallel_invalid_n_workers_raises(tmp_path: Path):
    pcfg = ParallelConfig(
        out_dir=tmp_path, n_workers=0, cell_label="smoke", config={},
    )
    try:
        run_parallel([1], random_game_worker, pcfg)
    except ValueError:
        return
    raise AssertionError("expected ValueError for n_workers=0")
