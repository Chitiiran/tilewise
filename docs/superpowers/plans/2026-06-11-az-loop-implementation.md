# Catan AZ Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline — stages babysit live WSL runs). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `mcts_study/catan_az/` — a resumable AlphaZero iteration loop (self-play → window buffer → train → arena gate → publish → journal) and run iteration 1 end-to-end on tonight's corpus.

**Architecture:** Thin orchestration over existing, proven pieces: `self_play_async.py` (self-play), `train_main` (training), AsyncMcts+BatchedGnnEvaluator (arena), bot_registry/web (ladder consumers). New code is glue + bookkeeping, all unit-testable without GPU.

**Tech Stack:** Python 3.12 (WSL venv), pytest, pandas/parquet, torch (only at the edges).

**Spec:** `docs/superpowers/specs/2026-06-11-az-loop-design.md`

---

### Task 1: AzConfig (config.py)

**Files:** Create `mcts_study/catan_az/__init__.py`, `mcts_study/catan_az/config.py`; Test `mcts_study/tests/test_az_config.py`

- [ ] Failing tests: defaults match spec §7; JSON round-trip (`to_json`/`from_json`) preserves all fields; unknown JSON keys rejected (typo guard).
- [ ] Implement frozen dataclass + round-trip helpers.
- [ ] Green + commit.

### Task 2: Sliding-window buffer (buffer.py)

**Files:** Create `mcts_study/catan_az/buffer.py`; Test `mcts_study/tests/test_az_buffer.py`

`select_window(iter_dirs: list[Path], window_games: int) -> list[Path]` — newest-first run dirs until the per-dir games.parquet row counts sum ≥ window_games; returns selected dirs (whole dirs — train_main takes run_dirs). Plus `count_games(run_dir)` reading games*.parquet row counts (sum across shards, skipped games excluded via winner != null guard if column exists).

- [ ] Failing tests on tmp parquet fixtures: window smaller than one dir → that dir only; spans dirs newest-first; all dirs when window exceeds total; empty dir skipped.
- [ ] Implement with pandas; no torch import.
- [ ] Green + commit.

### Task 3: Elo ladder + champion registry (ladder.py)

**Files:** Create `mcts_study/catan_az/ladder.py`; Test `mcts_study/tests/test_az_ladder.py`

State in `<loop_root>/ladder.json`: entries `{name, checkpoint, elo, games, created_iter}` + `champion` key. Elo: standard logistic, K=24, from an arena result (wins_a, wins_b, draws). `register_candidate`, `record_arena`, `promote(name)`, `champion()` helpers; atomic write (tmp+rename).

- [ ] Failing tests: fresh ladder seeds champion at 1000; arena 66%-34% moves ratings apart symmetrically; promote flips champion + appends history; atomic write survives simulated partial write (write tmp, crash before rename → old file intact).
- [ ] Implement.
- [ ] Green + commit.

### Task 4: Arena (arena.py)

**Files:** Create `mcts_study/catan_az/arena.py`; Test `mcts_study/tests/test_az_arena.py`

`run_arena(candidate_ckpt, champion_ckpt, cfg, out_dir) -> ArenaResult{wins_cand, wins_champ, draws, timeout_rate, per_rotation}`.
Seating: 2 candidate + 2 champion seats, 4 rotations of the base pattern [C,X,C,X], 30 seeds each (shared seed list both nets see identically). Async: one BatchedGnnEvaluator per net, AsyncMcts per seat, greedy (no Dirichlet, τ=0 — arena measures strength, mirrors AGZ eval mode). Reuses the e10e async game-driver pattern (chance fast-path, single-legal fast-path, step cap). done.txt resumable. Verdict helper `should_promote(result, cfg)` enforcing promote_threshold AND timeout_rate ≤ cfg max (else verdict "invalid").

- [ ] Failing tests (no GPU): rotation/seat assignment correct + every seed used once per rotation; `should_promote` boundary cases (55.0% exactly → hold; 55.1% → promote; great winrate but 6% timeouts → invalid); ArenaResult serialization.
- [ ] Implement; game-driver smoke-tested via 2 games at sims=2 on CPU in the integration test (Task 7), not unit tests.
- [ ] Green + commit.

### Task 5: Status + journal (status.py)

**Files:** Create `mcts_study/catan_az/status.py`; Test `mcts_study/tests/test_az_status.py`

`StatusWriter(loop_root)`: `stage(iter_n, stage_name, **fields)` → atomic status.json {iter, stage, ts, fields}; `journal_row(dict)` → append `journal.csv` (header on create, stable column order, missing keys empty).

- [ ] Failing tests: status overwrites atomically; journal appends + preserves header/columns across rows with differing keys.
- [ ] Implement + green + commit.

### Task 6: Orchestrator (loop.py)

**Files:** Create `mcts_study/catan_az/loop.py`; Test `mcts_study/tests/test_az_loop.py`

`run_iteration(cfg, loop_root, iter_n, *, selfplay_fn, train_fn, arena_fn)` — stage functions injected (defaults wrap self_play_async/train_main/run_arena) so the orchestrator is unit-testable with fakes. Stage order per spec §2; each stage writes status; skip-if-done markers (`<loop_root>/iter_<N>/STAGE.done`); on arena verdict promote→ladder.promote + checkpoint copy to `<loop_root>/checkpoints/az_iter_<N>.pt`, hold→journal only. `run_forever(cfg)` = while True: next iter, with KeyboardInterrupt-safe stop + `STOP` sentinel file check between stages.
CLI: `python -m catan_az.loop --loop-root ... --config ... [--iter N | --forever] [--skip-selfplay-dirs dir1 dir2]` — the skip flag is how iteration 1 consumes tonight's existing 5-proc corpus.

- [ ] Failing tests with fake stage fns: full iteration calls stages in order + writes journal; crash after train → rerun skips selfplay+train (done markers) and resumes at arena; STOP sentinel halts between stages; promote path updates ladder + copies checkpoint; hold path leaves champion.
- [ ] Implement + green + commit.

### Task 7: Micro-iteration integration test

**Files:** Test `mcts_study/tests/test_az_integration.py` (marked `slow`)

- [ ] One real iteration, tiny: games_per_iter=4, sims=8, n_concurrent=4, arena_games=8 (2/rotation), max_epochs=1, h32/L2 scratch net on CPU. Asserts: shards written, candidate checkpoint exists, arena verdict recorded, journal row complete. Budget ≤ ~10 min.
- [ ] Run, green, commit.

### Task 8: Iteration 1 live

- [ ] When 5-proc run completes (~06:42 cap or earlier): final corpus count + games/h into journal (cite).
- [ ] `python -m catan_az.loop --iter 1 --skip-selfplay-dirs <the 5 run dirs>` → TRAIN (window=all iteration-1 games) → ARENA (candidate vs Cell6 champion, 120 games) → verdict → journal + ladder.
- [ ] Journal entry + commit + PR update regardless of promote/hold (honest negatives are currency).

### Task 9: B1 inference-server spike (timeboxed)

- [ ] Spike script `scripts/spike_inference_server.py`: server proc (model on GPU, batches across client connections, window flush) + N client procs sending state_to_pyg-encoded eval requests in a tight loop. Measure round-trip p50/p95 + aggregate evals/s at N=10 vs in-proc baseline.
- [ ] Verdict vs go/no-go (≥2× aggregate games/h projection) → journal + spec addendum. Wire into self-play only if GO (separate task, next session if needed).

### Task 10: Ladder → web tier

- [ ] `bot_registry.list_difficulties()` gains dynamic `az-champion` entry when `<loop_root>/ladder.json` exists (label "AZ Champion (iter N)"). Test with tmp ladder.json. Commit.
