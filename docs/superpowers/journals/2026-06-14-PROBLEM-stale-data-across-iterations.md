# PROBLEM (to revisit): iterations reusing stale data + claims need a full audit

**Date:** 2026-06-14
**Status:** OPEN — recorded for later, not yet fixed. Loop left running.

## The suspected problem

While the champion stays the same (az_iter_1 through iters 2–5), the fresh-ratio
deficit can compute to **0**, so an iteration generates **no new self-play
games** and retrains on the *already-accumulated* pool. Effect: successive
"candidates" train on near-identical data → come out similar → near-50%
winrates + high draws. The loop spins fast (no 6h self-play) but may not be
truly AlphaZero-iterating (real AZ plays new games every iteration).

## What is VERIFIED against real files (2026-06-14)

`/home/chitii/catan_data/runs/v3/az_loop/iter_*/selfplay/`:
- **iter_3:** has self-play data in THREE bursts —
  - 7 dirs × ~110–120 games (~810 total), stat-created **05:01**
  - 7 dirs × 4 games (28), stat-created **06:06** (a resume top-up)
  - 7 dirs × 1 game (7), stat-created **07:12** (second resume top-up)
- **iter_4:** selfplay dir **EMPTY** — 0 new games.
- **iter_5:** selfplay dir **EMPTY** — 0 new games (in progress at time of check).

So: iters 4 and 5 generated **no new data** and trained on iter-3's pool.
That part of the earlier claim is confirmed.

## What was CLAIMED but is NOT yet trustworthy (needs audit)

1. **"21 az_iter_1 self-play dirs"** — `_all_selfplay_dirs` returned 21, but
   those are all iter-3's three-burst dirs, NOT spread across iterations as the
   phrasing implied. Re-verify exactly which dirs `_all_selfplay_dirs` /
   `fresh_deficit` actually count, per iteration.
2. **Timestamp mismatch (IMPORTANT):** dir NAMES say `T03-23-07`, `T09-14-47`,
   `T10-35-35` but `stat` mtimes say `05:01`, `06:06`, `07:12`. The
   name-vs-mtime gap (and the WSL-clock resyncs seen earlier this session)
   means the **sequence/attribution of which burst belongs to which run is not
   reliable** from a quick glance. A proper audit must reconcile: dir name
   (make_run_dir UTC stamp) vs mtime vs the daily_run*.log launch times vs the
   journal rows.
3. **"deficit = 0 so zero new data this iter"** — verified for iter-4/5 by the
   empty dirs, but the GENERAL claim (that this will keep happening) should be
   confirmed by reading `fresh_deficit` + `_all_selfplay_dirs` against the
   on-disk dirs carefully, not inferred.

## Why this matters

If true, the loop's recent HOLDs (iter-3/4/5) may be an artifact of retraining
on a fixed corpus, NOT evidence that improving on az_iter_1 is hard. The
winrate trend (37→44→51) could be noise on the same data rather than real
progress. We can't tell which until the data-flow is audited.

## To do when we come back

1. **Audit the data flow rigorously:** reconcile dir-name timestamps vs mtimes
   vs daily_run logs vs journal; confirm exactly how many NEW games each
   iteration generated and what each iteration's training window actually
   contained (`select_window` output per iter).
2. **Decide the fix** (candidates, not yet chosen):
   - `min_fresh_games_per_iter` — always generate ≥N new games even when
     deficit is 0, so each iteration sees new experience.
   - Generate self-play from the latest CANDIDATE (not just champion), so even
     held candidates evolve the pool.
   - Re-examine whether fresh-ratio-against-champion is the right policy when
     the champion is sticky.
3. **Cross-check the divergence-filter drop counts** per iter too (they hint at
   which corpus each iter loaded: iter-3 build dropped 204/225116, iter-4
   dropped 159/189051 — different corpus sizes, so the window DID differ
   between iters; reconcile that with "same pool" suspicion).

## Caveat on this very note

My earlier verbal claims this session about data generation were partly wrong
(the "21 dirs spread across iterations" framing) and the timestamps are
internally inconsistent. **Treat every quantitative claim here as provisional
until the audit above is done.** Recording now so we don't lose the thread.
