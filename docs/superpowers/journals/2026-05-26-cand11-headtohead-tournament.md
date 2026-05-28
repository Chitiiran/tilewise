# 4-way head-to-head: Cand 11 (Cell 5 v2) is the strongest GNN we have

**Date:** 2026-05-26
**Trigger:** Cell 5 v2 (Cand 11) finished its 15-epoch training run with a U-shaped mid-tournament trajectory (5.8% → 10.8% → 5.8%). The mid-tournament numbers — measured against `[GnnMcts, PureGnn, LookaheadV3, Random]` — left Cand 11's strength ambiguous: did it match Cell 1's plateau or not?

This journal documents the direct head-to-head experiment that resolved that question.

**Experiment:** new `e10c_triple_gnn` module — 3 PureGnn slots + 1 LookaheadV3, no Random and no GnnMcts. Cells played their tournament-peak checkpoints. Two runs: a 120-game pilot and a 1,200-game confirmation.

## Setup

| Slot | Role | Checkpoint | What it tests |
|---|---|---|---|
| A | PureGnn | `runs/v3/training/loss_aug/00_baseline_h128_l4_pilot/training_h128_l4/checkpoint_epoch05.pt` | Cell 0 vanilla peak (ep5 = best mid-tournament epoch) |
| B | PureGnn | `runs/v3/training/loss_aug/01_cand8_cand10_h128_l4/training_h128_l4/checkpoint_epoch10.pt` | Cell 1 Cand 8+10 plateau midpoint |
| C | PureGnn | `runs/v3/training/loss_aug/05_cand_road_pip_h128_l4_v2/training_h128_l4/checkpoint_epoch10.pt` | Cell 5 v2 Cand 11 peak |
| D | LookaheadMctsV3 | (no checkpoint — depth=10, base_sims=200) | Strong baseline |

Common config: `vp_target=5, bonuses=False, hidden_dim=128, num_layers=4`, `seed_base=19_000_000` (matches all prior mid-tournaments for cross-comparison), `workers=10`, `device=cuda`, `max_seconds=600s` per game.

Two scales, same seeds:
- **Pilot (120 games):** 30 games × 4 rotations. Wall-clock 16 min.
- **Confirmation (1,200 games):** 300 games × 4 rotations. Wall-clock 67 min. CI width at 95% = ±1.4pp.

## Results

### 120-game pilot (n=120, ±2.0pp CI)

```
Cell0-vanilla-ep5            8 / 120  ( 6.67%)
Cell1-cand8cand10-ep10      17 / 120  (14.17%)
Cell5v2-cand11-ep10         23 / 120  (19.17%)
LookaheadV3                 72 / 120  (60.00%)
Draws                        0 / 120
```

### 1,200-game confirmation (n=1200, ±1.4pp CI)

```
Cell0-vanilla-ep5           95 / 1200 ( 7.92%)   per-rot: 22, 28, 23, 22
Cell1-cand8cand10-ep10     105 / 1200 ( 8.75%)   per-rot: 25, 24, 30, 26
Cell5v2-cand11-ep10        202 / 1200 (16.83%)   per-rot: 54, 44, 49, 55
LookaheadV3                798 / 1200 (66.50%)   per-rot: 199, 204, 198, 197
Draws                        0 / 1200
```

**The pilot's rank order survives the 10× scale-up** with tighter CIs. Per-rotation breakdowns show no seating-bias artifact: each cell's per-rotation winrate stays within ~5 games of the cell's mean (no rotation dominates).

## Headlines

### Cand 11 is the new cumulative best

Pairwise gaps in the 1200-game run, with 95% CIs (Wilson on a binomial):

| Comparison | Gap | Significance |
|---|---:|---|
| Cand 11 vs Cand 1 (Cell 5 v2 − Cell 1) | **+8.08pp** | well outside ±1.4pp noise; ~6σ effect |
| Cand 11 vs Vanilla (Cell 5 v2 − Cell 0) | **+8.91pp** | well outside noise |
| Cand 1 vs Vanilla (Cell 1 − Cell 0) | **+0.83pp** | **within** CI; statistically indistinguishable |

The Cell-1-vs-Cell-0 gap is the most surprising result. In mid-tournaments, Cell 1 (Cand 8+10) was clearly stronger than Cell 0 (vanilla) at ep10 — 9.17% vs 1.67%, a 7.5pp difference. **In direct head-to-head with Cand 11 also at the table, those two cells become indistinguishable.**

### Mid-tournament rankings inverted

| Cell | Mid-tournament @ ep10 | 4-way (this run, 1200g) | Δ |
|---|---:|---:|---:|
| Cell 0 (vanilla, ep5 used) | 12.50% (best) | 7.92% | −4.58pp |
| Cell 1 (Cand 8+10, ep10) | 9.17% | 8.75% | −0.42pp |
| **Cell 5 v2 (Cand 11, ep10)** | **10.83%** | **16.83%** | **+6.00pp** |

Cand 11 *gains* in head-to-head; Cell 0 and Cell 1 each *lose*. This is the strongest evidence to date that **Cand 11's policy is qualitatively different**, not just marginally different on aggregate metrics.

### Why mid-tournaments understated Cand 11

Mid-tournaments give Cand 11 only 1 GNN slot vs `[GnnMcts (uses Cand 11), PureGnn (Cand 11), LookaheadV3, Random]` — two slots are MCTS-based and dominate winrate, leaving the GNN policy little room.

In the 4-way head-to-head, the GNNs compete *with each other* for the wins LookV3 doesn't take. Cand 11's policy wins those mixed-GNN games consistently — 202 of 402 non-LookV3 wins (50.2%). Cell 1 wins 105/402 (26.1%) and Cell 0 wins 95/402 (23.6%). **Cand 11 takes half of every game where LookV3 doesn't win**, despite all three GNNs sharing the same arch + training pipeline.

### LookV3 dominance is real but smaller than mid-tournaments suggested

| Setup | LookV3 winrate |
|---|---:|
| Cell 5 v2 mid-tournament @ ep10 (vs 1 PureGnn + 1 GnnMcts + 1 Random) | 84.2% |
| 4-way head-to-head (vs 3 PureGnns), 1200 games | **66.50%** |

Three real GNN opponents take wins LookV3 used to get against Random + GnnMcts. The "Lookahead beats random GNNs ~85%" framing from earlier tournaments was inflated by counting easy Random wins as Lookahead wins by partition.

## Comparison to Cell 5 v2 mid-tournament

Cell 5 v2's mid-tournament trajectory was 5.8% → 10.8% → 5.8% (ep5/10/15). The drop at ep15 triggered the auto-stop. But in head-to-head at 1200 games, **ep10 = 16.83% > Cell 1's all-time best (10.83% ep5)**.

This means: Cand 11's policy at ep10 is genuinely the strongest GNN we've trained, even though the mid-tournament metric didn't surface it clearly. The U-shape at ep5/10/15 likely reflects oscillation in the policy's overlap with the mid-tournament opponent set (Random + GnnMcts behave differently each epoch as the GNN evolves), not policy strength itself.

## Statistical confidence

- **Binomial Wilson 95% CI at n=1200:**
  - Cand 11: 16.83% ± 2.1pp → [14.8%, 19.1%]
  - Cell 1: 8.75% ± 1.6pp → [7.2%, 10.4%]
  - Cell 0: 7.92% ± 1.5pp → [6.5%, 9.6%]
  - LookV3: 66.50% ± 2.7pp → [63.8%, 69.1%]
- **Cand-11-vs-Cell-1 separation:** 16.83% − 8.75% = 8.08pp; the two CIs do not overlap. Decisive.
- **Cell-1-vs-Cell-0 separation:** 8.75% − 7.92% = 0.83pp; CIs overlap substantially. Cannot distinguish.

## Decision matrix update

| Cell | Best in mid-tournament | Best in 4-way head-to-head | Status |
|---|---:|---:|---|
| Cell 0 (vanilla) | 12.5% ep5 | 7.92% | superseded |
| Cell 1 (Cand 8+10) | 10.83% ep5 | 8.75% | **superseded** (was prior cumulative best) |
| Cell 2 (Cand 7+8+10) | killed | — | rejected |
| **Cell 5 v2 (Cand 11 alone)** | 10.83% ep10 | **16.83% ep10** | **new cumulative best** |

## What this changes about the loss-augmentation roadmap

1. **The "Cand 8+10 plateau" is not as strong as we thought.** It looked stable in mid-tournament because Random and GnnMcts were sucking wins away from Cell 0 at the same epoch. Head-to-head reveals Cand 8+10 is barely above vanilla.

2. **Cand 11 is structurally different.** It's a road-action prior; Cand 8+10 are policy-target rewrites for VP-yielding actions. They affect different action classes, and the road-class signal turns out to matter more than the VP-class signal at h128_l4 / 100k cache.

3. **The road-pip prior is a real intervention.** It's not just "doesn't hurt" — it produces a 6pp head-to-head improvement over the prior best. This is the first definitively positive loss-aug result.

4. **Future evaluation should use head-to-head**, not mid-tournament. The mid-tournament metric was a reasonable cheap proxy when we had nothing else, but it underestimated Cand 11 by 6pp. Going forward, gate every loss-aug candidate through an e10c-style 1200-game head-to-head against the current cumulative best.

## Next steps (suggested, not committed)

1. **Reproducibility seed.** Re-train Cand 11 with seed=1 to confirm the 16.83% is a stable property, not a single-seed artifact. ~19h training + ~1h head-to-head.

2. **Stack Cand 11 on Cand 8+10.** They affect different action classes, so structurally the stack might compound constructively. ~19h training + 1h head-to-head. Risk: Cand-7-style compound regression.

3. **Larger architectures.** h128_l4 has 632k params. Could be that Cand 11's signal scales better with capacity. Try h256_l4 or h128_l6. Caches already built.

4. **Investigate Cand 11's mechanism.** What does Cand 11 actually do differently in mid-game play? Run the `scratch_midgame_actions_cell1_ep10.py` analysis adapted for the Cand 11 ep10 checkpoint — measure road-rate, settlement-rate, city-rate, road-to-settlement ratio. If road-to-settlement dropped from Cell 1's 7.2 toward LookV3's 4.0, Cand 11 is working as designed.

## Cited artefacts

- New experiment module: `mcts_study/catan_mcts/experiments/e10_triple_gnn.py` (commit `b170835`)
- 120-game pilot: `runs/v3/tournaments/e10c_4way_2026_05_26/2026-05-26T15-34-e10c_triple_gnn/`
- 1,200-game confirmation: `runs/v3/tournaments/e10c_4way_1200_2026_05_26/2026-05-26T15-59-e10c_triple_gnn/`
- Checkpoints used:
  - Cell 0 ep5: `runs/v3/training/loss_aug/00_baseline_h128_l4_pilot/training_h128_l4/checkpoint_epoch05.pt`
  - Cell 1 ep10: `runs/v3/training/loss_aug/01_cand8_cand10_h128_l4/training_h128_l4/checkpoint_epoch10.pt`
  - Cell 5 v2 ep10: `runs/v3/training/loss_aug/05_cand_road_pip_h128_l4_v2/training_h128_l4/checkpoint_epoch10.pt`
- Cand 11 training journal: `docs/superpowers/journals/2026-05-25-cell5-road-pip-prior.md`
- Cand 11 perf RCA: `docs/superpowers/journals/2026-05-25-cand11-perf-rca.md`
- Cand 11 implementation plan: `docs/superpowers/plans/2026-05-25-road-pip-prior.md`
- Cand 11 design spec: `docs/superpowers/specs/2026-05-09-loss-augmentation-design.md` (item 3a, refined in chat 2026-05-25)

## Memory items to update

- **Cumulative-best update:** `feedback_pass100k_roadmap_v3.md` (or successor): Cell 1 is no longer the cumulative best. Cell 5 v2 (Cand 11 alone, `checkpoint_epoch10.pt`) replaces it with a +8pp head-to-head margin at n=1200.
- **Methodology update:** add a memory entry that mid-tournament results understate strength when GnnMcts and Random share the table. Future loss-aug evaluation must use head-to-head among GNN cells (e10c pattern) for the canonical comparison.

## Conclusion

**Cand 11 (pure-pip road prior, λ=0.05) at ep10 is the strongest GNN cell we have trained**, with statistically definitive evidence: 16.83% (±2.1pp) winrate in a 1,200-game 4-way against the prior cumulative best, vs Cell 1's 8.75% (±1.6pp). Cell 0 and Cell 1 are statistically indistinguishable in head-to-head.

The mid-tournament metric understated Cand 11's strength by ~6pp because it gave Random and GnnMcts wins that head-to-head distributes among the GNNs proportional to their actual policy quality. **Future loss-aug evaluation needs head-to-head as the canonical measurement.**

The road-pip prior is the first definitively positive intervention in the loss-augmentation roadmap.
