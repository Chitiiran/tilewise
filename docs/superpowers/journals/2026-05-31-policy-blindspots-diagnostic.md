# Where GnnMcts overrides PureGnn — the policy's blind spots

**Date:** 2026-05-31
**Goal:** before trying to fix PureGnn (which loses to LookV3 ~4% while GnnMcts
wins ~53%), DIAGNOSE exactly which decisions search corrects. Step (2) of the
user's "learn from GnnMcts: do 2 first then 1" — understand the blind spots, then
distill.
**Method:** `analyses/gnnmcts_vs_puregnn_divergence.py` replays all 80 Gate-2
games; at every GnnMcts-seat decision (>1 legal), compares the move MCTS actually
played vs what PureGnn (argmax of the SAME Cell6 policy head) would have played.
**5,580 GnnMcts decisions, 0 replay failures.**

## Headline: search overrides the policy on 60% of decisions

**3,357 / 5,580 (60.2%) of GnnMcts decisions DIVERGE from PureGnn's argmax.**
Search isn't making rare tactical corrections — it's overruling the raw policy on
the *majority* of moves. That's why GnnMcts (53.8%) so massively outperforms
PureGnn (6.2%): the policy is wrong most of the time, and search fixes it live.

## Policy blind spots — divergence rate by action type

How often does search override the policy on each move type? (high = blind spot)

| Action type | override rate | interpretation |
|---|---:|---|
| **trade_bank** | **83.6%** | policy almost never picks the right bank (4:1/port) trade |
| **propose_trade** | **79.0%** | policy badly mishandles player-to-player trades |
| play_yop (year-of-plenty) | 65.7% | wrong resources chosen |
| settle | 60.3% | wrong settlement vertex |
| robber | 60.1% | wrong robber target |
| road | 58.7% | wrong road placement |
| endturn | 56.9% | passes when it should act (or acts when it should pass) |
| play_roadbuilding | 47.8% | mistimed |
| play_mono | 45.2% | wrong monopoly resource |
| play_knight | 39.8% | mistimed knights |
| city | 16.2% | mostly right |
| **buy_dev** | **0.0%** | PERFECT — search never overrides dev-card buying |

## The smoking gun: trading + knowing when not to pass

**Of all the moves MCTS chose that the policy MISSED, 54.8% are trades**
(propose_trade 39.1% + trade_bank 15.7%). And the policy's single biggest WRONG
instinct is **endturn (33.3% of what PureGnn wrongly wanted)** — the raw policy
wants to *end the turn* when search knows there's a valuable trade or build to
make first.

So the policy's catastrophic weakness is **resource management**:
1. It can't trade (84% / 79% override on bank / player trades).
2. It passes prematurely (wants endturn when it should act).

This is the mechanism behind the earlier deep-analysis finding that PureGnn/RL
nets **stall at ~4.5 mean VP**: they build fine but can't convert resources via
trade, so they run out of the wood/brick/ore/wheat needed to keep building, and
just end their turns. Search finds the trades that unstick them.

## What the policy ALREADY knows

It's not uniformly bad — the diagnostic shows the policy has cleanly learned:
- **buy_dev (0% override)** — it knows exactly when to buy dev cards. (Consistent
  with Cell6's Cand-8 dev-card prior — that signal trained well.)
- **city (16% override)** — upgrades are mostly right.

So the net DID learn the build/dev economy. It just never learned to **trade** —
plausibly because trades are a large, sparse action space (40 trade actions:
20 bank + 20 propose) and the supervised targets rarely emphasized them, so the
policy head defaults to "endturn" over an unfamiliar trade.

## Implication for step (1) — distillation

The fix is now precisely targeted: distill GnnMcts's CHOSEN MOVES into the
policy, with the knowledge that **the high-value signal is in the trade and
endturn decisions** — the 55% of corrections that are trades, plus the
pass-vs-act timing. A distillation corpus that captures strong GnnMcts trade
behavior is the thing most likely to move PureGnn off ~4%.

Caveat / open question: trades may be a fundamentally hard thing for an argmax
policy to get right (the right trade depends on exact hand composition + what's
buildable, which is a sharp, combinatorial signal). If even a clean distillation
can't teach trading, that argues the **search IS the value** — GnnMcts is the
deployable player, and the policy is a fast-but-incomplete approximation. The
distillation experiment (step 1) will tell us which.

## Cited
- Script: `analyses/gnnmcts_vs_puregnn_divergence.py`
- Source: Gate-2 games `runs/v3/tournaments/gate2_e10e_async/`, net round0_Cell6
- Companion: `2026-05-31-gate2-clean-rerun.md`, `2026-05-31-three-tournament-deep-analysis.md`
- The "stall at ~4.5 VP" finding: the three-tournament deep analysis.
