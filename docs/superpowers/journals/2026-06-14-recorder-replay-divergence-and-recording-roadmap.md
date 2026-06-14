# Recorder↔replay divergence — filter (Path A) + full-observation recording roadmap

**Date:** 2026-06-14
**Status:** Path A shipped (build-time filter); root-cause + recording redesign tracked here for the next data-gen cycle.

## The bug (root cause, evidence-backed)

The dataset reconstructs each training position by **replaying** a self-play
game's stored `action_history` through the engine. For ~0.05% of positions this
replay diverges: the history reproduces **fewer gated decisions** for a player
than the recorder logged move rows.

Example (iter-3 corpus): seed 35000002, player 2 — recorder logged **104** move
rows (move_index 0..103), but replaying the stored history reproduces only **72**
gated player-2 decisions before diverging. Move rows mi=72..103 for that player
can never be replayed.

**Mechanism:** the self-play *generator* and the dataset *replay* disagree about
the game partway through — most likely a determinism difference in **chance-node
or robber-steal handling** (the two paths sample/apply those differently, so
after some point the replayed engine state diverges from generation and the
gated-decision attribution shifts). It was always latent; it first bit when the
corpus got large enough that a training run hit a divergent game.

## Path A — build-time filter (SHIPPED, commit f0ff30a)

`CatanReplayDataset.__init__` now replays each game **once** (per-game, not
per-position) via `replayable_decision_counts()` to get the replayable decision
ceiling per player, then `_drop_divergent_rows()` drops any move row whose
move_index exceeds it.

- iter-3: dropped **91 / 189,051 (0.048%)**; 0 failures in a 5000-sample after.
- Cost: one replay per game (~810), not per position (~189k) — fast.
- The earlier `resilient_getitem` per-item skip stays as a backstop, but the
  build-time filter is what actually unblocks training (the skip can't escape a
  contiguous block of >64 failing positions from one divergent game).

### Tracking for each data-gen cycle
The dataset logs `dropped N/M divergent positions (X%)` at build. **Watch this
number.** Expected ~0.05%. If it spikes (e.g. >1%), the divergence is getting
worse — likely because an engine-fidelity change widened the generator↔replay
gap — and is the trigger to prioritize the recording redesign below.

## Why replay-from-history won't scale — the real fix (NEXT CYCLE)

(User's insight, 2026-06-14.) Reconstructing positions by replaying
`action_history` is fragile **by construction**: it requires the engine to be
perfectly deterministic and identical between self-play generation and dataset
replay. As the engine grows toward real Catan, that gap only widens:

| Coming fidelity change | New divergence surface |
|---|---|
| Robber steal-victim as a learned decision | another chance/decision branch to replay identically |
| Strategic discard (player chooses cards) | per-card decision sequence to reproduce |
| Trade accept/reject + multi-resource trades | opponent-response branches, ordering sensitivity |
| Randomized board | board-gen must replay bit-identically from seed |

Each one is a new chance for replay to diverge from generation — i.e. the 0.05%
will grow. Patching replay determinism per-feature is a losing game.

### The redesign: record the full observation per move
Instead of storing `(seed, action_history)` and *regenerating* observations at
train time, **store the observation tensor (or the bytes to build it) directly
at generation time**, alongside the visit-count policy target and outcome.

Benefits:
- **No replay at all** → no recorder↔replay divergence, ever (eliminates this
  whole bug class).
- **Faster training** → replay is currently the CPU-bound cost of dataset build;
  recording removes it.
- **Fidelity-proof** → engine changes can't break already-recorded data; the
  observation is frozen at record time.

Costs / decisions to make next cycle:
- **Disk**: an observation is larger than a replay seed. Estimate: the v2
  observation is ~bit-packed; at ~hundreds of bytes/position × ~190k positions
  ≈ tens-to-hundreds of MB per iteration. Manageable, and the HDD-archive
  lifecycle already exists. (Measure exact size before committing.)
- **Schema**: bump `schema_version`; `CatanReplayDataset` gains a "recorded-obs"
  path that skips replay when the obs column is present, falling back to replay
  for old corpora.
- **Recorder change**: `SelfPlayRecorder` / `self_play_async` write the obs at
  each recorded move (the engine already exposes `observation()`).
- **Backward compat**: old replay-based corpora still load (replay path kept);
  new corpora use the recorded obs.

This is a focused, well-scoped next-cycle project. It should be done **before**
the engine-fidelity curriculum (robber/discard/trades) lands, since those are
exactly what would otherwise inflate the divergence rate.

## Decision log
- **Now:** Path A (filter) — unblock continual training on the clean 99.95%.
- **Next data-gen cycle:** full-observation recording (eliminates the bug class
  + speeds training + scales with fidelity). Tracked here.
- **Deferred:** root-causing the exact chance/robber determinism divergence — the
  recording redesign makes it moot, so don't spend hours on it unless the drop
  rate spikes before the redesign lands.
