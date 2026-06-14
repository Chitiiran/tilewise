# Iter-3 training crash → two resilience fixes (systematic debugging)

**Date:** 2026-06-14
**Trigger:** First real daily run completed 810-game self-play (6h, fresh-ratio
window worked) then **crashed in training** with:
`RuntimeError: Could not replay seed=33000048 to move_index=48 (player=0);
only saw 41 matching decisions.` — and the whole day's run exited.

## Root-cause investigation (the data is SOUND)

Worked the systematic-debugging phases. Every check exonerated the data:

| Check | Result |
|---|---|
| Did seed 33000048 time out? | No — clean game, winner=2, 592 moves, history len 592 |
| Replay the exact failing move in isolation | **20/20 deterministic 'ok' at move_index=48** |
| Build the real CatanReplayDataset, replay ALL its player-0 rows | **0 failures** |
| Sample 3000 positions across the whole 188,892-position corpus | **0 failures** |
| num_workers (fork corruption)? | 0 — single-process DataLoader, refuted |
| Run-dir collision / dup rows? | None — seed in exactly one dir, no dups |

**Conclusion: a rare, non-reproducible transient.** Best explanation: an
engine/fs glitch on one position during that run (`runs/v3` is a WSL-boundary
symlink that dropped earlier this session). Per the skill, this is a
well-investigated "no stable root cause" — chasing the ghost further is not the
fix.

## The real fix: defense in depth (don't let one bad position kill 6h)

The architectural lesson (the failure model, spec §2): **an environment/data
glitch on ONE position must be survived, never crash the loop.** Two fixes:

### 1. `resilient_getitem` in dataset.py (commit 98abe1b)
A position that can't be replayed is **skipped + logged**, substituting the next
valid neighbor so the batch stays full. Raises only if `max_tries` consecutive
positions all fail (genuine corruption). 5 tests. Training now tolerates the
rare bad sample instead of dying on it.

### 2. Resume incomplete iterations (commit 750e9de)
A second gap surfaced on restart: `_next_iter_number` jumped to max+1 even
though iter-3 had crashed mid-cycle (dir exists, no PUBLISH.done) — so the
first relaunch started a fresh **iter-4**, abandoning the 6h iter-3 corpus.
Fixed: an iteration without PUBLISH.done is **resumed** (self-play + train
salvaged via done-markers). 3 tests.

## Salvage

iter-3's 810 games were intact (SELFPLAY.done present). After both fixes the
run was resumed at **iter 3**: top-up self-play to the 840 fresh target, then
TRAIN retried with the resilient dataset (no longer crashes on the bad
position), then ARENA. The 6h of self-play was not lost.

## Lesson for continual training

This is exactly the "category 2 (environment) failures are physical reality,
survive them" principle in action. Two more rough edges hardened: the trainer
tolerates a bad position; the daily driver resumes a crashed iteration. Both had
tests that would have caught them — added now.
