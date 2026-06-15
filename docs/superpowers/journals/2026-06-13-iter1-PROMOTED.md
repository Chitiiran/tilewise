# AZ iteration 1 — PROMOTED (first successful iteration on this stack)

**Date:** 2026-06-13 (completed autonomously)

## Result

| metric | value |
|---|---|
| candidate (az_iter_1) wins | **78 / 120** |
| champion (cell6) wins | 42 / 120 |
| draws | 0 |
| **timeouts** | **0** |
| **candidate winrate** | **65.0%** (promote bar: >55%) |
| verdict | **PROMOTE** |
| Elo | cell6 1000 → 996.4; **az_iter_1 → 1003.6** |
| per-rotation cand wins | [20, 17, 23, 18] (consistent across seatings) |

**This is the first AZ iteration on this stack that produced a net beating its
parent.** Every May-31 attempt regressed (RL_iter1 15% vs Cell6 55.8%, the 9h
loop noise around ~6%). The difference: a larger, more diverse self-play corpus
(611 distinct full-Catan games / 156,631 deduped positions) + the value-
perspective-correct async stack + a clean arena gate.

Zero timeouts means the verdict is NOT wall-clock-censored (the e5 lesson) —
it's trustworthy.

## Method (per spec 2026-06-11-az-loop-design.md)

- SELF-PLAY: 611 games, GnnMcts@200 self-play with Dirichlet+temperature, full
  Catan, champion=cell6 (the overnight 5-proc run).
- BUFFER: window over both run dirs (156,631 deduped positions).
- TRAIN: warm-start cell6, lr=2e-4, early-stopped epoch 3 (best val_top1=0.382
  at ep1 — warm-started nets peak early, as expected).
- ARENA: GnnMcts@200 candidate vs champion, 120 games, 4 rotations, shared
  seeds, 0 timeouts.
- PUBLISH: az_iter_1 promoted; appears as the "AZ Champion" web difficulty tier.

## Three bugs found + fixed along the way (all committed, all tested)

1. **`active_game_count` sentinel** (20cffa1): arena never set it, so batches
   couldn't all-park-flush and the watchdog spammed `999999984`. Fixed +
   regression test.
2. **Per-game wall-clock cap** (a658cb1): game 120 (seed 30030029) crawled
   45min+ alone, holding the gather() hostage while the verdict sat decided.
   Added `arena_game_max_seconds=600`. (It finished naturally as the restart
   landed, so the cap didn't fire this time, but it prevents the class of
   stall.)
3. **PUBLISH idempotency** (54af17d): the restart-to-unblock re-ran the
   unguarded publish stage → double Elo + duplicate journal row. Gated PUBLISH
   behind its own done-marker; repaired the live ladder/journal by hand.

## Throughput reality (the real lever, unchanged)

The arena took ~6h for 120 games — 99% CPU / ~24% GPU, the documented
CPU-bound wall. This is why each iteration is slow. The **B1 inference-server
spike** (`scripts/spike_inference_server.py`, committed) is the next throughput
lever and should run on the now-idle GPU before iteration 2 — if it clears the
≥2× bar, both self-play and arena speed up together.

## Next (awaiting user where noted)

1. **B1 spike** on the idle GPU — autonomous, measurement-only.
2. **Iteration 2**: new self-play with az_iter_1 as champion (the loop's
   `--forever` or `--iter 2` does this). Each iteration ~6-8h at current
   throughput; B1 may cut that.
3. **Robber action-space fork** (journal 2026-06-11) — still needs the user's
   call (changes action space, forces retraining).

The loop works end-to-end and the bot is improving. The deliverable arc is on
track.
