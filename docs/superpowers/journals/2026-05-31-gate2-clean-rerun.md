# Gate 2 — value-fixed GNN+MCTS inverts the prior finding (and beats LookV3)

**Date:** 2026-05-31 (~07:42 EDT)
**Trigger:** Gate 2 of the batched-evaluator plan — the clean e10e re-run on the
async MCTS stack, with the value-perspective fix
[[project_gnn_value_perspective_bug_2026_05_30]]. The validation we built the
whole stack for and (initially) deferred. This is the un-confounded answer to
the question that started this whole line of work two sessions ago: **is GNN+MCTS
worse than PureGnn?**

## Result

`e10e_async`, 79 games (1 still finishing, won't move the picture), full Catan
(vp=10, bonuses), GnnMcts seat at sims=200, **0 timeouts**.

| Player | Wins | % of finished |
|---|---:|---:|
| **GnnMcts_Cell6** (value-fixed async MCTS) | **42** | **53.2%** |
| LookV3 (heuristic + MCTS) | 27 | 34.2% |
| PureGnn_Cell6 (raw policy) | 5 | 6.3% |
| PureGnn_Cell1 | 5 | 6.3% |

## The inversion — last session was WRONG, and we now know why

Last session (`2026-05-29`, the synchronous `e10e_gnn_mcts` with the BUGGY
evaluator), GnnMcts got **1 win (1.2%)** and we concluded *"GNN+MCTS is worse
than PureGnn."* That conclusion was an artifact of the **value-perspective bug**:
the GNN value head is ego-relative but the old evaluator (and OpenSpiel's MCTS)
indexed it as absolute-seat, poisoning the backed-up Q-values for every
non-mover node. The code reviewer flagged this during Task 4; we fixed it in the
async MCTS (rotate ego→absolute before backup, proven by test).

With the bug fixed:

| | buggy (2026-05-29) | value-fixed (2026-05-31) |
|---|---:|---:|
| GnnMcts winrate | 1.2% (1 win) | **53.2% (42 wins)** |
| GnnMcts vs PureGnn | 1 vs 8 (LOST) | **42 vs 5 (WON 8.4×)** |

**A ~44× improvement.** Search on top of the GNN doesn't hurt — it helps
enormously, the *opposite* of the prior conclusion.

## The bigger headline: first GNN player to beat LookV3

LookV3 (hand-tuned VP heuristic + MCTS) dominated EVERY prior tournament:
67–70% in the 4-quadrant matrix, and it beat every PureGnn cell decisively.
**GnnMcts_Cell6 beats it 53.2% to 34.2%** — the first time a GNN-based player
has won a tournament containing LookV3. The learned net + correct search is now
stronger than the engineered baseline.

## Why this matters for the whole effort

The batched-evaluator project's stated goal was throughput. Its *largest* payoff
turned out to be correctness: **catching the value-perspective bug in review
unlocked a fundamentally stronger player.** The original two-sessions-ago
question — "does search help the GNN?" — is now answered cleanly: **yes,
decisively, once the value head is indexed correctly.**

Practical consequences:
1. **Deploy GnnMcts, not PureGnn.** Search is worth ~47pp here.
2. **The old `gnn_evaluator.py` still has the bug.** Any future sync GnnMcts must
   use the same ego→absolute rotation, or just use the async stack.
3. **RL self-play with this MCTS is a strong teacher** — the value-fixed
   visit-count + value targets are now trustworthy, which is exactly what the
   RL loop needs. (RL iter-1 used sims=100 of this corrected MCTS; iter-2 in
   progress uses sims=160.)

## Caveat / scope

- n=79 games, single net pair (Cell6 as both PureGnn and GnnMcts), so the
  GnnMcts-vs-PureGnn comparison is clean (same weights, only search differs).
- The LookV3 result is the headline but n is modest; a larger re-run would
  tighten the CI. The effect size (53% vs 34% vs 6%) is far larger than
  sampling noise at n=79, so the ranking is secure even if exact percentages move.
- 0 timeouts: unlike the 2026-05-29 run (32% timeouts under GPU contention),
  this clean run has none — the result is not a wall-clock artifact.

## Cited
- `project_gnn_value_perspective_bug_2026_05_30` — the bug + fix.
- `project_e10e_gnnmcts_worse_than_puregnn_2026_05_29` — the prior (wrong) finding this overturns.
- Tournament: `runs/v3/tournaments/gate2_e10e_async/`
- Harness: `catan_mcts/experiments/e10e_async.py`
