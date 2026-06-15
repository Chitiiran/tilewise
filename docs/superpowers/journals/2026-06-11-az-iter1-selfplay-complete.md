# AZ iteration-1 self-play complete — PAUSED before train/arena (user instruction)

**Date:** 2026-06-11 (overnight run 00:42–06:46 WSL clock)
**Status:** corpus complete + validated; loop **paused until further notice** (user: "pause when the data is here. stop until further notice")

## Final corpus (all numbers measured)

| run dir | games (distinct seeds) | seed ranges | note |
|---|---|---|---|
| `distill/2026-06-11T00-42-self_play_async` | 144 | 21M | P1 (n_concurrent=64) |
| `distill/2026-06-11T00-46-self_play_async` | 467 | 22M:121, 23M:116, 24M:114, 25M:116 | P2–P5 **shared dir** (see bug) |
| **total** | **611** | | ~12 MB parquet |

All 5 procs ran to the full 6 h cap (`wall-clock cap 21600.0s hit` in every log).
Aggregate throughput: **611 games / 6 h ≈ 102 games/h** (early waves peaked ~150/h;
P1's 64-deep concurrency left its final in-flight wave unfinished at cap — only
144 of its games landed, vs ~117 each for the 24-deep procs. **Lesson: deep
concurrency trades completed-games-at-cap for batch efficiency; 24 was the
better setting.**)

Sample-validated earlier in the night: 127 decision positions in game 1,
policy normalized over legal actions, ego-perspective value vector correct,
loads in `CatanReplayDataset`.

## BUG found: run-dir collision + consolidation race (must fix before next multi-proc launch)

`make_run_dir` names dirs with **minute resolution** — P2–P5 launched within
the same minute and all wrote into `…T00-46-self_play_async`. Consequences,
verified by direct parquet inspection:

1. **Game rows duplicated exactly 2×** in the shared dir (934 rows / 467
   distinct seeds): the end-of-run `rec.checkpoint()` consolidations ran
   concurrently and absorbed each other's per-seed shards.
2. p4/p5 crashed with `FileNotFoundError` on per-seed shards mid-consolidation
   (another proc's checkpoint had just absorbed the file) — AFTER data was
   safe; cosmetic-but-alarming tracebacks.
3. **No data lost**: distinct seeds (611 incl. P1) match done.txt exactly;
   seed ranges were disjoint by design.

**Required fixes (when resuming):**
- Dedup at training time (or pre-pass): drop duplicate `seed` rows in games
  shards and duplicate `(seed, move_index, current_player)` rows in moves
  shards — otherwise the shared dir's positions are sampled at 2× weight.
- `make_run_dir`: add seconds + PID (or accept an explicit `--run-name`) so
  parallel launches can never collide.
- The loop's default self-play (one proc per iteration) doesn't hit this;
  any future multi-proc launch must pass distinct out dirs.

## State at pause

- `catan_az` loop built + tested (26 unit + 1 integration); ladder seeded
  (champion=cell6 @ 1000) at `/home/chitii/catan_data/runs/v3/az_loop`.
- Turnkey resume (DO NOT run without explicit user go): runbook
  `mcts_study/scripts/distill_runbook.md` §DIRECTION CHANGE — but apply the
  dedup pre-pass first.
- B1 spike script committed, not yet run (needs idle GPU).
- Monitors stopped; nothing running.
