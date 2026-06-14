# Candidate-Self-Play Redesign — Pilot Conclusion (2026-06-14)

**Goal of the pilot:** validate every mechanism of the candidate-self-play AZ
redesign end-to-end on live data, plus the live analytics dashboard, at tiny
scale (fast, watched) **before** committing to the multi-day full run.

**Config:** fresh `az_pilot/` root seeded from champion `az_iter_1`;
`games_per_iter=40`, `arena_games=30`, `arena_min_decisive=12`,
`max_iters_per_model=2`, `sims=200`, `vp=10 + bonuses`, 7 self-play workers.

---

## Mechanism scorecard

| Mechanism | Result | Evidence |
|---|---|---|
| `gen_iter` tagging in meta.json | ✅ | iter_1 `gen_iter=1 gen=az_iter_1`; iter_2 `gen_iter=2 gen=cand_iter_1` |
| **Latest-net self-play** (iter_2 self-plays with iter_1's *trained candidate*, not champion) | ✅ **core fix** | iter_2 workers ran `--checkpoint iter_1/training/checkpoint_best.pt`; meta `gen=cand_iter_1` |
| Stale-data fix: `new_games>0` in PROGRESS.md | ✅ | iter_1 row `new_games=35` (the bug's silent-zero column, now non-zero) |
| Quota counts only own `gen_iter` games (`own_iter_games`) | ✅ | generated 35 toward the 40 quota for gen_iter=1 only |
| Full cycle self-play→train→arena→verdict→journal | ✅ | iter_1 `hold` row in journal.csv (cand 11 / champ 11 / 6 draws / 50%) |
| Training health (early-stop, checkpoints) | ✅ | iter_1 early-stop ep2 val_top1=0.505; iter_2 ep2 val_top1=0.454 |
| Dashboard liveness across ALL stages | ✅ | self-play via `daily_state.json` mtime fallback; train/arena via `status.json` ts; both LIVE, no false flags |
| Dashboard verdict + Elo render | ✅ | `/api/metrics` served iter_1 verdict=hold elo=1000.0 |
| Graceful stop at `max_iters` | ✅ (by design) | `run_day` loops `while done < max_iters` (=2); also `holds>=max_iters_per_model` guard |
| **Window-reset-on-promotion** | ⚠️ **NOT exercised** | both iters HOLD (50% < 65% bar) → no promotion → reset path never triggered. Unit-tested only (`ladder.promote(promoted_at_iter=)` + `last_promotion_iter()`). |

**Verdict on the redesign:** every mechanism whose absence caused the
2026-06-14 stale-data bug is now present and demonstrated on live data. The one
unvalidated end-to-end path (window-reset) only fires on a promotion, which a
2-iter pilot from a converged start is not expected to produce; it has unit
coverage. **The redesign is sound.**

## Bug caught by the pilot (fixed)

**Dashboard liveness read the wrong file** (commit `3024b4b`). `run_cycle`
writes `daily_state.json` atomically at every stage, but `status.json` only at
stage *transitions* — so during the hours-long self-play stage the dashboard
showed "NOT RUNNING" while 7 workers ran at 95% CPU. Fixed `liveness()` to fall
back to `daily_state.json` mtime when `status.json` has no usable ts. This is
the dashboard's single most important answer ("is my 6-day run alive?") and it
was wrong for the first ~hour of every iteration. +2 tests, 25 green. Verified
live (selfplay→LIVE, arena→LIVE).

## Gaps still open before the full run

1. **Per-game flush (RECOMMEND FIX FIRST).** Self-play writes games.parquet
   only when a worker finishes its whole `--num-games` batch (`p.wait()` then
   write, daily.py:133). At sims=200, ~6 min/game → a crash mid-batch loses
   ALL of that worker's games. In the pilot a worker held 5 games for ~30 min
   before flushing. On the full run (`1000//7≈142` games/worker, **many
   hours** unflushed) a single WSL hiccup / OOM / power event loses an entire
   worker's iteration of compute. This directly violates the standing
   salvage-compute rule (per-game persistence for runs >30 min). **This is the
   one gap worth fixing before launch.**

2. **Integer-division game shortfall (cosmetic).** `per = n_games // n_procs`
   drops the remainder: `40//7×7=35`, `1000//7×7=994`. The deficit-resume loop
   (`own_iter_games` on rerun) tops it up, so the quota is eventually met; a
   single clean pass just under-produces by `n_games % n_procs`. Low priority.

3. **`git: not a repository` warning (harmless).** The training subprocess
   tries to stamp a git SHA via a broken nested-worktree path
   (`.../az-bots/C:/dojo/...`, a Windows-path-in-Linux artifact). Non-fatal —
   training completes immediately after. The SHA just isn't recorded. Cosmetic.

## Full-run readiness

**Mechanically ready.** Recommended pre-launch: fix gap #1 (per-game flush) so
the multi-day run can survive a crash without losing a worker's hours of
compute. Gaps #2/#3 are cosmetic and can ride along.

Full-run scale (per the redesign spec): `games_per_iter=1000`,
`arena_games=300`, `promote_threshold=0.65`, `max_iters_per_model=10`;
~14h/iter, ~6 days for 10 iters. Launch remains gated on explicit user go.
