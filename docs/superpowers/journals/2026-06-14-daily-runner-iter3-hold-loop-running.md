# Daily runner works end-to-end — iter-3 HOLD, loop self-continuing

**Date:** 2026-06-14
**Milestone:** the faithful-AZ daily runner ran a full cycle
(self-play → train → arena → publish) **autonomously** and **auto-continued**
to the next iteration — the continual-training goal, achieved.

## iter-3 verdict: HOLD (clean, valid)

| iter | champion | window_dirs | cand | champ | draws | draw% | cand winrate (decisive) | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | cell6 | 2 | 78 | 42 | 0 | 0% | 65% | promote |
| 2 | az_iter_1 | 7 | 23 | 39 | 58 | 48% | 37% | **invalid** |
| 3 | az_iter_1 | 13 | 38 | 49 | 33 | 27.5% | **43.7%** | **hold** |

Candidate az_iter_3 (Elo 1002.5) lost narrowly to champion az_iter_1 (1004.7).
Champion holds.

### Why this is progress despite a HOLD
- **Valid, not invalid.** iter-2 was *invalid* (48% draws). iter-3's draw rate
  fell to **27.5%** — the VP-margin tiebreak is working, so the gate produced a
  trustworthy verdict instead of a no-signal mess.
- **Fresh-ratio window worked.** 840 fresh current-champion games, no dilution
  (the iter-2 root cause is fixed). The candidate simply wasn't *better*.
- **Improving on az_iter_1 is genuinely hard.** It's already a decent net; AZ
  gains per iteration are small and noisy at this game-count scale. The gate
  correctly declining a 43.7% candidate is the gate doing its job.

## The loop is self-running (and self-limiting)

After publishing iter-3, `run_day` automatically started **iter-4** (training as
of this writing). The divergence filter logged `dropped 159/189051` on iter-4's
build — stable ~0.08%, consistent with the tracking table. No crashes.

**Stagnation guard:** iter-2 (invalid) + iter-3 (hold) = 2 trailing
non-promotes. At `stagnation_holds=5` the runner stops + flags. So the loop will
either find an improving candidate or halt itself at 5 — it won't grind blindly.

## Strategic question (for the user — not an error)

Repeated narrow HOLDs against az_iter_1 raise a tuning question for *continual*
training:
1. **More games/iteration** — 120-game arenas are noisy near 50%; a real ~3pp
   improvement needs more games to clear the 55% bar with confidence. (Costs
   more compute/iteration.)
2. **Accept many iterations** — AZ improves over *many* small steps; a few
   HOLDs early is expected. Let the loop run; the anchor-vs-LookV3 match (every
   5 iters) gives absolute progress signal even during relative plateaus.
3. **Promote-by-margin / lower bar** — relax the 55% gate, or promote on a
   smaller-but-statistically-significant edge. (Risks promoting noise.)

My lean: **(2) let it run** — the stagnation guard bounds the downside, and the
anchor match will show whether absolute strength is creeping up even while
relative promotions stall. Revisit (1) if the anchor is flat after several iters.

## Bugs fixed to get here (this session)
recorder↔replay divergence (build-time filter) · resilient_getitem skip ·
resume-incomplete-iteration · VP-margin tiebreak. All committed + tested.

## What's running
iter-4 training now; loop continues autonomously. Persistent monitor watches the
verdict. The divergence-recording redesign (record full observation, stop
replaying) is the tracked next-cycle change.
