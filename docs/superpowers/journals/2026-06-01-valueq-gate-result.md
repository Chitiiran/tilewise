# ValueQ (1-ply value-Q) gate result — TIES raw argmax, does NOT fix the plateau

**Date:** 2026-06-01
**Hypothesis (from the plateau diagnosis):** PureGnn's argmax-of-policy discards
the value information that distinguishes Catan's near-equal moves. A 1-ply
value-Q deployment (score each legal child by the proven-fitting value head, pick
the mover's-value argmax) should recover that and beat raw argmax.

## Result — 120-game gate (e10f_valueq_async, all seats = Cell6 net)

| player | wins | appearances | win%/appearance |
|---|---:|---:|---:|
| LookV3 | 71 | 120 | **59.2%** |
| **ValueQ** | 17 | 120 | **14.2%** |
| RawPureGnn | 32 | 240 | **13.3%** |

mean_batch=10.5, 0 skipped/timeout, shared seeds, seat-rotated.

**ValueQ 14.2% vs RawPureGnn 13.3% — a statistical TIE.** The 1-ply value-Q
deployment does NOT beat raw argmax. The gate FAILS.

(The 8-game smoke showed 37.5% vs 6.2% — that was pure small-sample noise. n=8
is uninformative; this is why we ran 120.)

## Why 1-ply value-Q didn't help (interpretation)
- The value head fits the TRAINING distribution (terminal-ish self-play states,
  D1). But 1-ply lookahead evaluates CHILDREN — often mid-turn, off-distribution
  states (right after a build, before end-turn) where the value head is less
  calibrated. So the Q estimates it ranks by are noisy exactly where it matters.
- 1-ply can't see the opponent's response. Catan's near-equal moves differ in
  what they ENABLE next turn; a single greedy value-step is blind to that.
- This is precisely why multi-ply SEARCH (tree statistics, opponent replies)
  recovers value where a single value-step does not. It sharpens the diagnosis:
  the gap is not "argmax vs value-argmax", it's "no-search vs search".

## What this means for the fix menu
- Option 2 (1-ply value-Q): **REJECTED by data.** Tie with argmax.
- Option 1 (cheap fixed-width search, sims=16): now the key test. It is REAL
  search (multi-ply, opponent replies), just cheap. The diagnosis predicts this
  is where the lift lives. Most likely to actually beat raw argmax + approach
  LookV3 at a fraction of GnnMcts cost.
- Option 3 (policy+value blend): even weaker than 1-ply value-Q; deprioritize.

## Caveat on cross-harness comparison
RawPureGnn scores 13.3% here vs ~5% in older e10e runs. Win% is FIELD-dependent
(opponents + seeds differ across harnesses), so absolute numbers don't transfer.
The ValueQ-vs-RawPureGnn comparison WITHIN this run is apples-to-apples (same net,
same seeds, same opponents) and is the valid read: a tie.

## Next
Build cheap fixed-width search (sims=16, PUCT, no tree reuse) as a deploy bot and
run the same 120-game gate. If it beats RawPureGnn and nears LookV3, that's the
shippable search-free* fix (*tiny search, not full GnnMcts).
