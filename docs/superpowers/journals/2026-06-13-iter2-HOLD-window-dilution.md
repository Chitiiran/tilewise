# AZ iteration 2 — HOLD (invalid): window dilution + a decision point

**Date:** 2026-06-13 (autonomous)
**Verdict:** iteration 2 did NOT promote. Champion stays **az_iter_1** (1003.6).

## Result

| iter | champion | window_dirs | cand | champ | draws | draw% | cand winrate (decisive) | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | cell6 | 2 | 78 | 42 | 0 | 0% | 65% | **promote** |
| 2 | az_iter_1 | 7 | 23 | 39 | 58 | 48% | **37%** | **invalid** (draw>40%) |

The gate worked perfectly: it declined to promote a candidate that was *worse*
than the champion (37% of decisive games) and produced mostly ties (48%). The
publish-idempotency fix held — a single clean journal row, no double-count.

## Why iteration 2 didn't improve (diagnosis)

**Window dilution.** The 237,906-position training window spanned 7 run dirs:
iter-2's 5 dirs (361 fresh az_iter_1 self-play games) **+ iter-1's 2 dirs (611
older cell6-era games)**. So ~63% of the training signal came from the *weaker,
older* policy. Warm-starting az_iter_1 and then training mostly on cell6-era
games pulled the candidate **backward**, not forward — it couldn't beat the
champion it started from.

Corroborating signals:
- **val_top1 rose to 0.430** (from iter-1's 0.382) yet winrate *fell* — a
  textbook confirmation of the standing hard rule that **val_top1 is not a
  winrate proxy**. Better label-fit on a diluted distribution ≠ stronger play.
- **48% draws** — candidate ≈ az_iter_1 (near-identical, as expected from
  warm-start + diluted signal).

## The fix for iteration 3 (a decision point — surfaced to user)

The sliding window (`window_games=1200`) is too large for the current
per-iteration self-play volume (~361 games), so it reaches back into the
previous champion's era. Canonical AZ wants the window dominated by *recent*
(current-champion) self-play. Options:

1. **More fresh self-play per iteration** (e.g. 800-1200 games of az_iter_1
   self-play) so the window is mostly current-champion games. Cost: ~6-8h of
   self-play per iteration at current throughput. Cleanest; most canonical.
2. **Shrink `window_games`** (e.g. to ~400) so even 361 fresh games dominate.
   Cheap (no new compute) but a smaller corpus = noisier training.
3. **Both**: moderate window (~600) + moderate fresh self-play (~600).

My recommendation: **option 1** for a real signal, but it's a compute/time
call the user should weigh (each iteration is a half-day). A quick way to test
the hypothesis cheaply first: **re-train iter-2's candidate on ONLY the 361
fresh dirs** (drop iter-1) and re-gate — if it then beats az_iter_1, window
dilution is confirmed as the cause and option 1/2 is the fix.

## What's solid

- Loop is proven: iter-1 promoted (65%), iter-2 correctly held. The gate
  protects the champion and now evaluates similar nets (VP-tiebreak).
- 4 bugs found+fixed this session (dedup, active_game_count, wall-clock cap,
  publish idempotency) + 1 methodology fix (VP-tiebreak) — all tested.
- B1 throughput: NO-GO (independent procs are best).

## Status: paused for a user call on iteration-3 window strategy

Not launching iteration 3 autonomously — it's a compute/time + parameter
decision (window size vs fresh-game volume) that the user should set, and
re-running the same way would likely HOLD again. The loop is ready to continue
the moment that's decided.
