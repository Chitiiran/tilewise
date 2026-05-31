# RL policy-iteration loop — iterate until LookV3 is beaten

**Date:** 2026-05-31 (start ~10:30 EDT)
**Goal (user):** "generate 100 games and iterate the RL process, continue until
Lookahead is beaten."

## Target (decided with user)

Iterate the RL loop; each round arena-test the new net **as PureGnn AND as
GnnMcts** against LookV3. **Stop when EITHER beats LookV3 convincingly.**
- GnnMcts already beats LookV3 at Cell6 (53.8%, Gate 2) — so the GnnMcts target
  may be met immediately by the best net; the loop's job is to find the net whose
  GnnMcts is *strongest* vs LookV3, and to push PureGnn toward beating LookV3
  (the hard, never-achieved target).
- Adaptive sims: start 160; raise (→400) if winrate plateaus across rounds.

## What iter-1/iter-2 got wrong (fixed here)

1. **Not true policy iteration.** iter-1/iter-2 both generated self-play from
   *Cell6* and trained from *Cell6*. AlphaZero generates from the LATEST net each
   round. **Fix:** round N self-play + warm-start use the round-(N−1) BEST net
   (round 0 best = Cell6).
2. **Arena was vs Cell6, not vs LookV3.** **Fix:** arena = `e10e_async` with
   LookV3 in the lineup, measuring the new net both ways vs LookV3.
3. **Tiny corpus.** iter-1=66, iter-2=193. **Fix:** ≥100 games/round (user said
   100), via parallel single-exec workers; accumulate across rounds (the corpus
   grows — round N trains on rounds 1..N if helpful, or just round N).

## Round structure

```
best ← Cell6 (round 0)
loop round = 1, 2, ...:
  1. SELF-PLAY: ~100 games from `best`, full Catan, sims=S (parallel workers).
  2. TRAIN: warm-start from `best`, on this round's data (+ optionally prior),
     early-stop, lr 5e-4, rotate → candidate net (checkpoint_best).
  3. ARENA: e10e_async, 80 games full Catan:
       A = candidate PureGnn,  B = candidate GnnMcts,
       C = best PureGnn (reference),  D = LookV3.
     Measure: candidate_PureGnn vs LookV3, candidate_GnnMcts vs LookV3,
              candidate vs best.
  4. PROMOTE: if candidate beats `best` (head-to-head or higher vs LookV3),
     best ← candidate.
  5. STOP if candidate_PureGnn > LookV3 OR candidate_GnnMcts > LookV3 by a clear
     margin (>~5pp over 80 games). Else continue.
  6. If no improvement for 2 rounds, raise sims (stronger teacher).
```

## Stopping criteria

- **PRIMARY WIN:** candidate **PureGnn** beats LookV3 head-to-head (the hard
  target — no PureGnn has ever done this).
- **PRACTICAL WIN:** candidate **GnnMcts** beats LookV3 by a margin clearly above
  Cell6's existing 53.8% (i.e. RL produced a stronger-than-Cell6 search player).
- **PLATEAU:** if 3+ rounds pass with no net improvement even after raising sims,
  stop and report the honest ceiling + what would be needed (bigger nets, far
  more games, MCTS-in-self-play tuning).

## Budget / throughput

- Self-play ~73 s/game/worker at sims=160; 5 parallel workers ≈ 100 games in
  ~25-35 min. Train ~5 min. Arena (e10e_async, GnnMcts seat is slow) ~40-60 min.
  → ~1.5-2 h/round. Each round committed + journaled.
- Per-game persistence; harness-tracked single-exec launches (NOT nohup/&+wait).

## Honest expectation

The deep analysis showed RL nets stall at ~4.5 mean VP and even Cell6-PureGnn
loses to LookV3 (6% vs 34%). Beating LookV3 with PureGnn is a HIGH bar that may
take many rounds or may not be reachable at this net size / data budget. The
GnnMcts target is likely already met by the best net; the loop will confirm
which net's GnnMcts is strongest. Each round is a committed milestone; the loop
reports honestly whether it's converging toward the bar or plateauing.
