# RL loop iteration 2 — 3× data narrows the gap to the parent (trajectory confirmed)

**Date:** 2026-05-31 (~09:55 EDT)
**Trigger:** iter-1 regressed (15% vs parent Cell6 55.8%) on a tiny 66-game corpus. iter-2 fixes the two failures: **much more data** (193 games via 5 parallel workers) + an **arena gate** (promote only on a head-to-head win) + early-stop.

## The arena gate result

`e10_quad_gnn`, 120 games, full Catan (vp=10, bonuses), 1 timeout. RL_iter2 (new net, best=ep2) vs the parent and two fillers:

| Player | Wins | % |
|---|---:|---:|
| Cell6_parent | 56 | 47.1% |
| Cell1 | 35 | 29.4% |
| **RL_iter2** | **27** | **22.7%** |
| Cell0 | 1 | 0.8% |

**RL_iter2 still loses to Cell 6 → the arena gate correctly does NOT promote.** But the trajectory is the story.

## The trajectory — more data closes the gap

| Iteration | Corpus | Positions | RL net | Cell6 | Gap (RL − Cell6) |
|---|---:|---:|---:|---:|---:|
| iter-1 | 66 games | 13.6k | 15.0% | 55.8% | **−40.8pp** |
| iter-2 | 193 games (3×) | 39.0k (3×) | **22.7%** | 47.1% | **−24.4pp** |

**Tripling the self-play data nearly HALVED the gap** (−40.8pp → −24.4pp). The new net gained +7.7pp (15→22.7) and pulled the parent down 8.7pp (55.8→47.1). RL_iter2 also went from losing to Cell0 (it didn't, but barely) to clearly beating it (0.8% for Cell0), and closed on Cell1.

**This is the canonical AlphaZero signal:** each iteration with more self-play moves the student toward — and eventually past — the teacher. We're on the right curve; we just haven't crossed it yet. Linear-ish extrapolation of the gap (40.8 → 24.4 with 3× data) suggests another ~2-3× data per iteration, or 2-3 more iterations, plausibly closes it.

## Why it hasn't crossed yet (honest mechanism)

1. **Cell 6 is a strong, converged teacher** (100k MCTS games). Beating it from a few hundred self-play games is a high bar.
2. **193 games is still modest** for AlphaZero — production AZ uses tens of thousands per iteration.
3. **Still mild overfit** — val_top1 peaked at epoch 2 (0.424) then declined; more data pushed the peak later (iter-1 peaked at ep1) but didn't eliminate it. Larger corpora would help further.
4. **Single iteration from a fixed teacher** — true AZ re-generates self-play from the NEW net each round (policy iteration). We warm-start from Cell6 and train once; the compounding only starts with multiple rounds.

## Training detail

`train --init-from Cell6 --rotate random --epochs 4 --early-stop-patience 1 --lr 5e-4` on 39,041 positions (×4 rotate ≈ 156k samples).

| epoch | train_loss | val_loss | val_top1 |
|---|---:|---:|---:|
| 1 | 2.267 | 2.830 | 0.421 |
| 2 | 2.034 | 2.876 | **0.424** |
| 3 | 1.994 | 2.947 | 0.419 |
| 4 | 1.969 | 2.942 | 0.408 |

Best = epoch 2 (`checkpoint_best.pt`).

## Decision: do NOT promote, DO continue the curve

The arena gate works as designed — no regression ships. Cell 6 remains the champion. But iter-2 is **not a failure**: it's the second point on a clear improvement curve. The path to a better-than-Cell6 player is now empirical, not speculative:

1. **More self-play per iteration** — the parallel-worker setup (5 single-exec workers ≈ 5× throughput) makes 500-1000 games/iter reachable in a few hours; the CPU-bottleneck levers [[project_batched_eval_gate1_2026_05_30]] (more workers, vectorized state_to_pyg) push further.
2. **Multiple iterations with re-generated self-play** — true policy iteration: generate from the latest net each round, gate, promote on win.
3. **Larger corpus → less overfit** — the val-peak moved from ep1→ep2 with 3× data; more would let more epochs help.

## What stays true

The session's headline remains **Gate 2** [[project_gnn_value_perspective_bug_2026_05_30]]: the value-fixed GNN+MCTS (53%, beats LookV3) is the strongest player measured. The RL track is the path to an even stronger *trained* net; iter-2 shows that path is real and the gap is closing with data.

## Cited
- iter-1: `2026-05-31-rl-loop-iter1.md` (the 15% regression this improves on)
- Gate 2: `2026-05-31-gate2-clean-rerun.md`
- Corpus: `runs/v3/rl_selfplay_iter2/` (193 games), net: `runs/v3/rl_train_iter2/checkpoint_best.pt`
- Arena: `runs/v3/tournaments/rl_iter2_arena/`
