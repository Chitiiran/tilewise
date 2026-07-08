# 2026-07-08 — Phase 1: batched arena GATE PASSED (20.1 min for 40 games @ sims=200)

**Context:** Phase 1 of `docs/superpowers/plans/2026-07-08-distillation-first-roadmap.md`,
executed task-by-task per `docs/superpowers/plans/2026-07-08-phase1-batched-arena.md`
(subagent-driven, fresh implementer + reviewer per task; ledger in `.superpowers/sdd/progress.md`).

## The headline measurement (Task 5 production gate)

```
GATE_RESULT games=40 seconds=1204.6 gpm=1.992
wins_cand=13 wins_champ=27 draws=0 timeouts=0
```

| Config | Throughput | 40-game wall-clock | Source |
|---|---|---|---|
| OLD un-batched arena, sims=100 | never finished | **>3 h, killed, no verdict** | shake-out journal 2026-06-19 §5 |
| OLD B=1 path, sims=200 (est.) | ~0.20 g/min | ~3.3 h | measured B=1 CUDA rate, journal 2026-06-18 |
| **NEW batched arena, sims=200** | **1.99 g/min** | **20.1 min** | this gate, measured |

- **Phase-1 exit gate (≤1 h): PASSED with 3x headroom.** ~10x speedup vs the estimated
  B=1 rate at equal sims; the 300-game production arena drops from ~25 h to ~2.5 h.
- Measurement caveats: the timed span includes batched-`.ts` export-if-missing, but this
  run cache-hit the export written by the smoke run (same checkpoint, same b_max) — the
  1204.6 s is arena work, not export. GPU observed engaged mid-run (36% util, ~20 W,
  driver ~97% CPU — consistent with two-queue leaf batching sharing one device).
- Sanity note: cand/champ were the SAME net (az_iter_1 vs itself). The 13–27 split is a
  ~2% binomial tail at n=40 — noted, not alarming (bit-exact reproducibility + 8/8 oracle
  agreement below say the machinery is fair); the Phase-2 120-game re-anchor is the
  proper strength measurement.

## What was built (commits 38cc010..HEAD on az-difficulty-bots)

1. **Task 1 — `ArenaSlot`** (`a9f2b55`): pausable per-game arena state (MtRng game chance,
   per-seat NpRngs seed+11/+13, greedy pick), proven FULL-GAME EQUAL to the B=1 oracle
   when driven at B=1 (4/4 rot×seed pairs).
2. **Task 2 — two-queue scheduler** (`1e2fb77`): `play_arena_games_batched` routes each
   game's parked leaf to its mover's net queue (cand/champ), flushes each at b_max.
   Bit-exact reproducible across runs; **8/8 winner agreement vs the B=1 oracle** (no
   float-reassoc argmax flips observed). `SearchSession` untouched (it was already
   net-agnostic — the seam analysis's key finding).
3. **Task 3 — PyO3 entry** (`dc6e399` + review fix `8fa316d`): `run_arena_games` gains
   all-or-nothing batched kwargs (partial set raises ValueError); shared
   `arena_result_to_dict` helper; B=1 serial path preserved byte-identical as the oracle.
4. **Task 4 — arena.py wiring** (`26c1888`): `_ts_batched` exports device-suffixed
   `.{dev}.b{max_batch}.batch.ts` once before the chunk loop; batched kwargs passed on
   every chunk; results.jsonl/dedup/ts-injection/PAUSE semantics unchanged.
5. **Task 5 — gate script** (`3b43f8b`): `mcts_study/scripts/arena_throughput_gate.sh`,
   GPU env taken by CALLING `_rust_cuda_env()` (drift-proof by construction), parseable
   GATE_RESULT line. Smoke-verified, then this production run.
6. **Task 6 — BUG B, SHA stamping** (`bb94715`): `_git_sha()` fallback chain (AZ_GIT_SHA
   env → git → textual worktree parse with Windows→WSL path translation → "unknown"+
   warning, never silently empty); `daily.py` stamps once at driver start and propagates
   to all workers. Live-verified from WSL against this real worktree (exact match to
   native `git rev-parse HEAD`). Bonus fix: upward `.git` search.
7. **Task 7 — observability minimums**: in flight at journal time (data-quality summary +
   degeneracy gate; val_value_mse + val_value_sign_acc beside val_top1).

Also this session (pre-Phase-1): suite-drift repair (`aaff407`) — 3 stale tests/fixtures
from the June-18/19 sessions fixed, zero engine bugs; new `test_rust_resume_deterministic_dir_no_offset`
pins the C2 pause/resume contract the 10k distillation teacher run depends on.

## Review-process notes

Every task passed an independent spec+quality review; Task 3 needed one fix round
(duplicated dict-building; silent partial-kwargs fallback → now ValueError). Minor
findings parked for the final whole-branch review are listed in the ledger.

## Next

- Task 7 review → final whole-branch review → PR `az-difficulty-bots` → main
  (user-approved 2026-07-08, review before merge).
- Phase 2: 120-game shared-seed re-anchor (GnnMcts@200 / RawPureGnn / LookV3) on the new
  arena — the comparability baseline for the distillation pipeline.
