"""Safety harness for the M1/M2/M3 refactor pyramid.

Runs end-to-end checks BEFORE and AFTER each refactor to catch
regressions in correctness, performance, or game outputs:

  1. Rust test suite             — engine semantics
  2. Python test suite           — pipeline integration
  3. Engine-replay regression    — 5 baseline games reproduce identically
  4. Pure-Rust perf floor        — bench_random_game no >1.5x slower
  5. Pure-Rust microbench floor  — bench_v1 no >1.5x slower
  6. Playback render             — one game renders without errors

Usage:
  cd mcts_study
  python scripts/safety_check.py [--quick]    # --quick skips perf checks

Exits 0 on green, 1 on any failure with a per-check summary.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).parent.parent  # mcts_study/
ROOT = HERE.parent  # engine-v2/
BASELINES = HERE / "tests" / "baselines"


def run(cmd: list[str], cwd: Path, timeout: int = 600) -> tuple[bool, str, str]:
    """Run cmd, return (ok, stdout, stderr)."""
    try:
        p = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired as e:
        return False, "", f"timeout after {timeout}s\n{e}"
    return p.returncode == 0, p.stdout, p.stderr


def parse_bench_random(stdout: str) -> dict:
    """Pull workload metrics out of bench_random_game JSONL stdout."""
    out = {}
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        out[d.get("workload", "?")] = d
    return out


def parse_bench_v1(stdout: str) -> dict:
    out = {}
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        out[d.get("workload", "?")] = d
    return out


def check(name: str, ok: bool, detail: str = "") -> None:
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name}{(' — ' + detail) if detail else ''}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="skip perf benchmarks (correctness + replay only)")
    args = ap.parse_args()

    failures: list[str] = []

    # 1. Rust test suite
    print("== 1. Rust test suite ==")
    ok, _, err = run(["cargo", "test", "--quiet"], cwd=ROOT / "catan_engine", timeout=600)
    check("cargo test", ok, "" if ok else err.splitlines()[-1] if err else "see logs")
    if not ok:
        failures.append("cargo test")

    # 2. Python test suite (excluding the one slow test we know takes 2 min)
    print("== 2. Python test suite ==")
    ok, _, err = run(
        [sys.executable, "-m", "pytest",
         "tests/test_engine_regression.py",
         "tests/test_engine_combined_calls.py",  # M1 fast-path tests
         "tests/test_recorder.py", "tests/test_recorder_schema_v2.py",
         "tests/test_dataset.py", "tests/test_state_to_pyg.py",
         "tests/test_gnn_model.py", "tests/test_gnn_evaluator.py",
         "tests/test_smoke_imports.py", "tests/test_experiments_common.py",
         "tests/test_evaluator.py", "tests/test_parallel.py",
         "tests/test_train.py",
         "-q", "--tb=line"],
        cwd=HERE, timeout=900,
    )
    check("pytest", ok, "" if ok else "tail: " + err.splitlines()[-1] if err else "see logs")
    if not ok:
        failures.append("pytest")

    if args.quick:
        print(f"\n== quick mode: skipped perf checks ==")
        if failures:
            print(f"\n{len(failures)} failure(s): {', '.join(failures)}")
            return 1
        print("\nALL GREEN")
        return 0

    # 3. Pure-Rust perf floor (bench_random_game)
    print("== 3. Pure-Rust random-game bench ==")
    floor = json.loads((BASELINES / "perf_floor.json").read_text())
    ok_run, stdout, err = run(
        ["cargo", "run", "--release", "--bin", "bench_random_game", "--",
         "--n-games", "100", "--seed-base", "800001", "--max-steps", "30000"],
        cwd=ROOT / "catan_engine", timeout=300,
    )
    if not ok_run:
        check("bench_random_game", False, "build/run failed")
        failures.append("bench_random_game (build)")
    else:
        bench = parse_bench_random(stdout)
        summary = bench.get("rand-game-summary", {})
        floor_game = floor["rand_game"]
        cur_us = summary.get("mean_us_per_game", 0)
        floor_us = floor_game["mean_us_per_game"]
        tol = floor_game["tolerance_factor"]
        ok_perf = cur_us <= floor_us * tol
        check("rand_game.mean_us_per_game",
              ok_perf,
              f"{cur_us:.0f} us (floor {floor_us:.0f}, max {floor_us * tol:.0f})")
        if not ok_perf:
            failures.append("rand_game perf")

        legal = bench.get("rand-legal_actions", {}).get("mean_ns", 0)
        floor_legal = floor["rand_legal_actions_ns"]["value"]
        tol_legal = floor["rand_legal_actions_ns"]["tolerance_factor"]
        ok_legal = legal <= floor_legal * tol_legal
        check("rand_legal_actions_ns",
              ok_legal,
              f"{legal:.0f} ns (floor {floor_legal:.0f}, max {floor_legal * tol_legal:.0f})")
        if not ok_legal:
            failures.append("rand_legal_actions perf")

    # 4. Microbench floor (bench_v1)
    print("== 4. Microbench (bench_v1) ==")
    ok_run, stdout, err = run(
        ["cargo", "run", "--release", "--bin", "bench_v1", "--",
         "--version", "safety-check"],
        cwd=ROOT / "catan_engine", timeout=300,
    )
    if not ok_run:
        check("bench_v1", False, "build/run failed")
        failures.append("bench_v1 (build)")
    else:
        bench = parse_bench_v1(stdout)
        floor_v1 = floor["bench_v1"]
        tol = floor_v1["tolerance_factor"]
        for key, floor_key in [
            ("bench-engine-step", "engine_step_us"),
            ("bench-mcts-game", "mcts_game_us"),
            ("bench-evaluator-leaf", "evaluator_leaf_us"),
            ("bench-state-clone", "state_clone_us"),
            ("bench-legal-mask", "legal_mask_us"),
        ]:
            cur = bench.get(key, {}).get("mean_us", 0)
            floor_v = floor_v1[floor_key]
            ok_b = cur <= floor_v * tol
            check(f"{key}.mean_us", ok_b,
                  f"{cur:.3f} us (floor {floor_v:.3f}, max {floor_v * tol:.3f})")
            if not ok_b:
                failures.append(f"{key} perf")

    # 5. Playback smoke render
    print("== 5. Playback smoke ==")
    out_dir = HERE / "runs" / "v2_smoke_postfix"
    candidates = list(out_dir.glob("*/games.sims=25.parquet"))
    if not candidates:
        check("playback smoke", False, f"no parquet under {out_dir}")
        failures.append("playback smoke (no parquet)")
    else:
        run_dir = candidates[0].parent
        seed = 725001
        ok_pb, _, err = run(
            [sys.executable, "-m", "catan_mcts.playback", str(run_dir), str(seed),
             "--out-dir", str(HERE / "runs" / "v2_smoke_postfix" / "_safety_render")],
            cwd=HERE, timeout=120,
        )
        check("playback render", ok_pb,
              "" if ok_pb else "tail: " + (err.splitlines()[-1] if err else "see logs"))
        if not ok_pb:
            failures.append("playback render")

    # Summary
    print()
    if failures:
        print(f"!! {len(failures)} FAILURE(S): {', '.join(failures)}")
        return 1
    print("ALL GREEN")
    return 0


if __name__ == "__main__":
    sys.exit(main())
