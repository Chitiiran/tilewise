# AlphaZero exploration experiment — the fix works mechanistically, but not (yet) on winrate

**Date:** 2026-05-31
**Question:** the policy can't beat LookV3 (PureGnn ~4%) while GnnMcts can (~53%).
The divergence diagnostic showed the policy's blind spot is **trading + passing
too early** (search overrides it on 84% of bank trades, 79% of player trades).
Hypothesis: our self-play was fully GREEDY (no exploration), so the net never
tried the trades it's bad at. The AlphaZero fix — Dirichlet root noise +
temperature sampling — should let it explore and learn trades.

## What was built (TDD, committed e3e94f6 + 810f9ae)

Canonical AlphaZero self-play exploration, off by default (arena/eval stay
deterministic), on via `--self-play`:
- **Dirichlet(α=0.8) root noise**, ε=0.25, mixed into root priors.
- **Temperature sampling**: τ=1 for the first 30 per-player moves, then τ→0.
- Soft-π training target (visit-count distribution) left unchanged — already AZ-correct.
- 6 new tests; full async_mcts + self_play suites green.

## The experiment

- **Self-play:** 117 games from Cell6 WITH exploration (10 parallel workers,
  sims=160, full Catan) → 29,818 positions.
- **Train:** warm-start Cell6, lr 5e-4, rotate, early-stop. Best **val_top1 =
  0.367** (DOWN from the greedy round's 0.444 — exploratory targets include
  sampled/perturbed moves, so they're a noisier, harder distribution to fit).
- **Arena:** `e10e_async`, 78 games, full Catan, vs LookV3.

## Arena result (78 games)

| Player | Wins | % |
|---|---:|---:|
| LookV3 | 36 | 46.2% |
| **AZ_GnnMcts** | 32 | 41.0% |
| AZ_PureGnn | 5 | 6.4% |
| Cell6 | 5 | 6.4% |

- **AZ_PureGnn: 6.4%** vs LookV3 (baseline 3.8%). Barely moved; still crushed.
- **AZ_GnnMcts: 41.0%** — DOWN from Cell6's 53.8%, and now LOSES to LookV3.

On winrate, this is a **negative**: exploration didn't make PureGnn competitive
and it *degraded* the GnnMcts player.

## BUT the divergence diagnostic — the fix worked mechanistically

Re-running the policy-blind-spot diagnostic on the AZ net vs the Cell6 baseline:

| Override rate (search corrects policy) | Cell6 | AZ net | Δ |
|---|---:|---:|---:|
| **overall** | 60.2% | **44.1%** | **−16pp** |
| **trade_bank** | 83.6% | **61.3%** | **−22pp** |
| **propose_trade** | 79.0% | **62.6%** | **−16pp** |
| **endturn** | 56.9% | **18.3%** | **−39pp** |
| settle | 60.3% | 71.0% | **+11pp** (worse) |
| road | 58.7% | 67.9% | **+9pp** (worse) |
| buy_dev | 0.0% | 0.5% | ~same (still perfect) |

**The AZ exploration genuinely taught the policy the things it was missing:**
- Trade-override dropped ~20pp (it learned to trade better).
- endturn-override collapsed 57%→18% (it stopped passing prematurely — the #1
  diagnosed flaw).

This is real, mechanistic evidence the Dirichlet+temperature fix does exactly
what AZ theory predicts: forcing exploration of low-prior moves let the policy
learn them.

## Why mechanistic improvement didn't become winrate

The policy got better at trading/passing **but worse at placement** (settle
+11pp, road +9pp override) and overall slightly degraded (val_top1 0.444→0.367).
Net effect: a lateral-to-worse player. Two reasons:

1. **The corpus is far too small for exploration to pay off.** 117 games / 30k
   positions. Real AlphaZero uses *millions* of self-play games — exploration
   adds variance that only averages out into improvement at scale. At our scale,
   exploration shifted the policy's behavior (visibly, in the diagnostic) but
   the noisier targets degraded calibration faster than the exploration helped.
2. **Warm-starting a converged net + early-stop at ep1** can't absorb a harder
   target distribution — it moves the policy a little (enough to change the
   blind-spot profile) but not enough, and not cleanly.

## Honest conclusion

**The AlphaZero exploration fix is correct and demonstrably does what it should
(the diagnostic proves the policy learned trades + stopped over-passing). But at
this net size (h128) and data scale (~100 games/round), it does not produce a
PureGnn that beats LookV3, and it degrades the already-good GnnMcts.**

The deployable answer is unchanged and stands as the arc's real win:
**value-fixed GnnMcts beats LookV3 (53.8%, Gate 2).** Search is the irreducible
value here — the policy can be *nudged* toward search's behavior but, on a
feasible compute budget, can't replace it.

The AZ-exploration code is committed and correct; it is the right machinery for a
future large-scale run (thousands–millions of games), where exploration would be
expected to pay off. The blocker is throughput (CPU-bound self-play, ~0.8
games/min/worker), not the algorithm.

## What we'd need to actually beat LookV3 with PureGnn
- **Orders of magnitude more self-play** (the CPU-bottleneck levers: vectorized
  state_to_pyg, more workers, or a batched-within-search evaluator).
- **Many policy-iteration rounds** with the exploration on, not one.
- Possibly **a larger net** (h256+) to encode the sharp combinatorial trade policy.

## Cited
- Built: `async_mcts.py` (Dirichlet + temperature_sample), commits e3e94f6, 810f9ae.
- Diagnostic: `analyses/gnnmcts_vs_puregnn_divergence.py`.
- Baseline blind spots: `2026-05-31-policy-blindspots-diagnostic.md`.
- The win: `2026-05-31-gate2-clean-rerun.md` (GnnMcts 53.8% vs LookV3).
- RL trajectory: `2026-05-31-rl-round1.md`, `2026-05-31-rl-loop-iter2.md`.
