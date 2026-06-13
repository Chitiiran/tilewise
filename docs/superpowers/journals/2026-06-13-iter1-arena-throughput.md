# Iteration-1 arena: bug fix + throughput decision

**Date:** 2026-06-13 (autonomous operation while user away)

## What happened

Iteration-1 TRAIN completed cleanly: warm-start cell6 on the 156,631-position
deduped corpus, early-stopped at epoch 3 (best val_top1=0.382 at epoch 1 —
consistent with the May-31 lesson that warm-started nets peak early), candidate
saved to `iter_1/training/checkpoint_best.pt`.

The ARENA stage then stalled: 98% CPU / 32% GPU, watchdog spamming
`999999984 live game(s) not parked`, 0 results.

## Bug found + fixed (committed 20cffa1)

`run_arena` created two `BatchedGnnEvaluator`s but never set their
`active_game_count`, so each kept the `10**9` default. Consequences:
- the all-parked flush clause (`n >= active_game_count`) could never fire →
  batches degraded to window-only (5ms) flushing;
- the watchdog (`n_before < active_game_count`) fired every window → the
  `999999984` (= 10**9 minus a few parked) spam.

Fix: a shared live-game counter sets both evaluators' `active_game_count` on
coroutine enter/exit (mirrors `self_play_async`'s `active` dict), with a
regression test that fails if the sentinel is ever left in place. 7 arena
tests pass. Restarted the loop — done-markers skipped SELFPLAY+TRAIN, resumed
straight into ARENA on fixed code.

## Throughput reality (measured, not estimated)

After the fix, the arena runs **correctly but slowly**: first valid result
(`cand` won, not timed out) landed at **~31 min** with 16 concurrent games.
99.7% CPU / ~24% GPU confirms the documented CPU-bound bottleneck
(state_to_pyg + tree management dominate; the GPU is starved). Projected:
120 games ≈ **4-5 hours**.

## Decision (autonomous, reversible): let it run

Per the user's standing "resources previously given" + "decide autonomously":
- The arena is **progressing correctly** (valid result, fix confirmed).
- It is **bounded** (200k-step per-game cap → stragglers mark timed_out, no
  infinite hang) and **resumable** (per-game jsonl).
- 4-5h fits the overnight budget; killing to reconfigure would waste the
  invested compute for marginal savings.
- Reducing arena scope is reversible (could rerun 120 later), so this is a
  safe best-judgment call, not a blocking decision.

## The real lever for FUTURE iterations: B1 inference-server spike

The CPU-bound wall is exactly what the B1 spike (committed
`scripts/spike_inference_server.py`) was scoped to measure. It should run on an
idle GPU (i.e. between iterations, not during this arena). If it clears the
≥2× bar, future self-play AND arena throughput improve together. This is the
right place to spend the throughput-engineering budget — not in preempting a
working gate.

## Next

- Let arena finish → iteration-1 verdict (promote/hold) + first Elo datapoint.
- Then B1 spike on the freed GPU.
- The robber-action-space fork (journal 2026-06-11) still awaits the user.
