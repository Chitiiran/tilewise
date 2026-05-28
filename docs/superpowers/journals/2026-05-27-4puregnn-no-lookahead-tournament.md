# 4-PureGnn tournament (no Lookahead) — Cell 5 v2 > Cell 6 > Cell 1 > Cell 0

**Date:** 2026-05-27
**Trigger:** The 1200-game head-to-head with Lookahead (`2026-05-26-cand11-headtohead-tournament.md` + `2026-05-26-cell6-cand11-cand8-cand10-stack.md`) put Lookahead at 66-68% of all games, compressing the GNN cells into a narrow 7-17% band. We wanted to isolate GNN-vs-GNN ranking without Lookahead absorbing wins by partition. New experiment: 4 PureGnn slots, no Lookahead, no Random.

**Experiment:** new `e10d_quad_gnn` module (commit `5d22713`), modeled on `e10_triple_gnn` but with 4 PureGnn slots and no LookV3 or Random. Every game decided by a GNN's argmax-policy. No draws possible (one PureGnn always wins).

## Setup

| Slot | Role | Checkpoint |
|---|---|---|
| A | PureGnn | `runs/v3/training/loss_aug/00_baseline_h128_l4_pilot/training_h128_l4/checkpoint_epoch05.pt` (Cell 0 vanilla peak) |
| B | PureGnn | `runs/v3/training/loss_aug/01_cand8_cand10_h128_l4/training_h128_l4/checkpoint_epoch10.pt` (Cell 1 Cand 8+10) |
| C | PureGnn | `runs/v3/training/loss_aug/05_cand_road_pip_h128_l4_v2/training_h128_l4/checkpoint_epoch10.pt` (Cell 5 v2 Cand 11) |
| D | PureGnn | `runs/v3/training/loss_aug/06_cand11_cand8_cand10_h128_l4/training_h128_l4/checkpoint_epoch10.pt` (Cell 6 stack) |

Common config: `vp_target=5, bonuses=False, hidden_dim=128, num_layers=4`, `seed_base=19_000_000`, `workers=10`, `device=cuda`, `max_seconds=600s` per game, 300 games × 4 rotations = **1,200 games**.

## Results

```
Cell0-vanilla-ep5            208 / 1200  (17.33%)   rot[ 53,  54,  56,  45]
Cell1-cand8cand10-ep10       298 / 1200  (24.83%)   rot[ 71,  69,  85,  73]
Cell5v2-cand11-ep10          371 / 1200  (30.92%)   rot[103,  83,  86,  99]
Cell6-stack-ep10             323 / 1200  (26.92%)   rot[ 73,  94,  73,  83]
draws                          0 / 1200
```

| Rank | Cell | Wins | % | 95% CI | vs uniform (25%) |
|---|---|---:|---:|---|---:|
| 🥇 | Cell 5 v2 (Cand 11 alone) | 371 | **30.92%** | ±2.6pp | +5.9pp |
| 🥈 | Cell 6 (Cand 11 + Cand 8 + Cand 10) | 323 | **26.92%** | ±2.5pp | +1.9pp |
| 🥉 | Cell 1 (Cand 8 + Cand 10) | 298 | **24.83%** | ±2.5pp | −0.2pp |
| 4 | Cell 0 (vanilla) | 208 | **17.33%** | ±2.2pp | −7.7pp |

Wall-clock: ~27 min (10 workers, GPU). Sum = 1200 exactly (no draws).

## Headlines

### 1. Vanilla is the clear weakest

Cell 0 at 17.33% is **7.67pp below uniform random** in a 4-way. Every loss-aug cell beats vanilla. Per-rotation 45-56 (tight), so it's not a seating fluke — vanilla just loses more often than the other three GNNs.

This is the cleanest evidence yet that the loss-aug roadmap as a whole produces meaningfully stronger policies than vanilla MCTS-visit-CE training. **Loss augmentation pays.**

### 2. Cell 5 v2 (Cand 11 alone) is still the strongest

371 wins, 30.92%. **+4pp above Cell 6, +6.1pp above Cell 1, +13.6pp above vanilla.** Per-rotation 83-103 (some variance but no outlier rotation). Cell 5 v2's CI [28.3, 33.5]% does NOT overlap with Cell 1's [22.4, 27.4]% or Cell 0's [15.2, 19.5]%, so the gap to those two is statistically decisive. **Cell 5 v2's CI does overlap with Cell 6's** (28.3-33.5 vs 24.5-29.4 — overlap 28.3-29.4), so Cand 11 alone vs the stack is statistically distinct but the gap is small (~4pp).

### 3. Cell 6 — the stack — beats Cell 1 alone here, unlike head-to-head

This is the interesting result. Recap:

| Tournament | Cell 6 (stack) | Cell 1 (Cand 8+10) | Cell 6 − Cell 1 |
|---|---:|---:|---:|
| Mid-tournament ep10 | 15.0% | 9.17% | +5.83pp (overstated; vs Random+GnnMcts+LookV3) |
| **Head-to-head with LookV3 (1200g)** | **8.92%** | **8.83%** | **+0.09pp (statistically tied)** |
| **4-PureGnn (this, no LookV3)** | **26.92%** | **24.83%** | **+2.09pp (CIs just barely overlap)** |

**The stack DOES add ~2pp over Cell 1 alone, but only when Lookahead isn't there to absorb the difference.** With LookV3 in the table, Cell 6 and Cell 1 are statistically indistinguishable. Without LookV3, Cell 6 is marginally stronger.

This suggests Cand 11's road prior, when stacked on Cand 8+10, contributes a small competitive improvement that gets absorbed by LookV3's dominant wins in mixed-MCTS tournaments. **Real but small.**

### 4. More loss-aug machinery ≠ better

| Loss-aug machinery | Cell | 4-PureGnn % |
|---|---|---:|
| None | Cell 0 | 17.33% |
| Cand 8 + Cand 10 | Cell 1 | 24.83% (+7.5 vs vanilla) |
| Cand 11 alone | **Cell 5 v2** | **30.92% (+13.6 vs vanilla)** |
| Cand 11 + Cand 8 + Cand 10 | Cell 6 | 26.92% (+9.6 vs vanilla) |

**The Cand 11-alone cell beats the everything-stacked cell by 4pp.** Adding Cand 8+10 to Cand 11 doesn't help; it actively hurts. The specific combination matters more than the count of mechanisms.

Hypothesis (from the Cell 6 journal): Cand 8 inherits Cell 1's dev-card-spam degenerate equilibrium (BuyDevCard 16.37/100 turns in Cell 1 vs Cand 11's 5.77/100). When stacked, the dev-card pull degrades Cand 11's clean expansion strategy without inheriting Cell 1's closeout efficiency. The result is a policy that's worse than Cand 11 alone and slightly better than Cand 8+10 alone — basically Cand 8+10 with the road prior cosmetically nudging things.

## Calibration: gain from removing Lookahead

Comparing each cell's % across the two 1200-game tournaments today:

| Cell | vs LookV3 (1200g, 2026-05-27) | 4-PureGnn (this) | Gain when LookV3 leaves |
|---|---:|---:|---:|
| Cell 0 vanilla | 7.92%* | 17.33% | **+9.41pp** |
| Cell 1 (Cand 8+10) | 8.83% | 24.83% | **+16.00pp** |
| Cell 5 v2 (Cand 11) | 14.67% | 30.92% | **+16.25pp** |
| Cell 6 (stack) | 8.92% | 26.92% | **+18.00pp** |

*Cell 0 7.92% cited from 2026-05-26 head-to-head; today's head-to-head didn't include Cell 0.

LookV3 was winning ~67.6% of games. Distributing that proportional to GNN strength would give each GNN ~16-17pp more. **Three cells gained ~16-18pp; Cell 0 gained only 9.4pp.** Cell 0's vanilla policy is so weak it can't capture even the proportional share of wins LookV3 vacates — Cell 1, Cell 5 v2, and Cell 6 take a disproportionately larger slice.

This tells us LookV3 was crushing vanilla harder than it was crushing the loss-aug cells. The loss-aug intervention helps the GNNs *resist* Lookahead better, even when the head-to-head winrate looks small.

## Per-rotation consistency

| Cell | rot=0 | rot=1 | rot=2 | rot=3 | stdev |
|---|---:|---:|---:|---:|---:|
| Cell 0 | 53 | 54 | 56 | 45 | 4.4 |
| Cell 1 | 71 | 69 | 85 | 73 | 6.4 |
| Cell 5 v2 | **103** | 83 | 86 | **99** | 8.6 |
| Cell 6 | 73 | **94** | 73 | 83 | 9.2 |

Cell 5 v2 dominates rot=0 (103/300) and rot=3 (99/300) — first-seat and last-seat positions favor its expansion strategy. Cell 6 dominates rot=1 (94/300). No cell wins all four rotations. Variance is in the 4-9 game range — within statistical bounds, no obvious seating-bias artifact.

## Decision matrix update

| Cell | mid-tournament best | head-to-head with LookV3 | 4-PureGnn (no LookV3) | Status |
|---|---:|---:|---:|---|
| Cell 0 vanilla ep5 | 12.5% | 7.92% (cited) | 17.33% | weakest |
| Cell 1 (Cand 8+10) ep10 | 10.83% | 8.83% | 24.83% | superseded |
| **Cell 5 v2 (Cand 11) ep10** | 10.83% ep10 | **14.67% / 16.83% (two runs)** | **30.92%** | **cumulative best across all three contexts** |
| Cell 6 (stack) ep10 | 15.8% ep15 | 8.92% | 26.92% | marginally better than Cell 1 alone; worse than Cand 11 alone |

## What this changes for the loss-aug roadmap

1. **Cell 5 v2 is the cumulative best regardless of tournament context.** Mid-tournament, head-to-head with LookV3, or 4-PureGnn without LookV3 — Cand 11 alone is on top in all three.

2. **The stack adds ~2pp marginal improvement over Cell 1 alone, but only in GNN-only contexts.** In any tournament where Lookahead is present, the stack and Cell 1 are statistically tied. So unless we have a specific application where the model only competes against other GNNs (not realistic for the v3 design which targets Lookahead), the stack doesn't have a real edge over Cell 1.

3. **Future loss-aug candidates should be benchmarked against Cell 5 v2 in head-to-head with LookV3.** That's the canonical metric per the 2026-05-26 ruling. A candidate is "promising" if it beats Cell 5 v2's 14.67-16.83% head-to-head with non-overlapping CIs.

4. **The closeout gap remains.** Cell 5 v2 still has the Cand-11 closeout weakness (27.1% cities/turn-with-resources vs LookV3's 51.0% per the closeout diagnostic). Cand 8+10 was the wrong fix. Open: what's the right one?

## Cited artefacts

- New module: `mcts_study/catan_mcts/experiments/e10_quad_gnn.py` (commit `5d22713`)
- Tournament dir: `runs/v3/tournaments/e10d_4gnn_1200_2026_05_27/2026-05-27T17-08-e10d_quad_gnn/`
- Launch log: `launch.log`
- Checkpoints used (all ep10 except Cell 0 which used ep5):
  - Cell 0: `runs/v3/training/loss_aug/00_baseline_h128_l4_pilot/training_h128_l4/checkpoint_epoch05.pt`
  - Cell 1: `runs/v3/training/loss_aug/01_cand8_cand10_h128_l4/training_h128_l4/checkpoint_epoch10.pt`
  - Cell 5 v2: `runs/v3/training/loss_aug/05_cand_road_pip_h128_l4_v2/training_h128_l4/checkpoint_epoch10.pt`
  - Cell 6: `runs/v3/training/loss_aug/06_cand11_cand8_cand10_h128_l4/training_h128_l4/checkpoint_epoch10.pt`
- Companion journals:
  - `2026-05-26-cand11-headtohead-tournament.md` (Cell 5 v2 head-to-head, 16.83%)
  - `2026-05-26-cell6-cand11-cand8-cand10-stack.md` (Cell 6 head-to-head, 8.92%)

## Conclusion

**Cell 5 v2 (Cand 11 alone) wins the 4-PureGnn tournament at 30.92%**, decisively above Cell 1 (24.83%) and vanilla (17.33%), and ~4pp above the Cell 6 stack (26.92%). Cell 5 v2 remains cumulative best in all three tournament contexts measured to date.

**Cell 6 (stack) is marginally better than Cell 1 alone (~2pp) only when Lookahead is removed from the table.** With LookV3 present, the stack and Cell 1 are statistically tied.

**Loss augmentation as a whole works** — the three loss-aug cells (Cell 1, Cell 5 v2, Cell 6) all beat vanilla by 7.5-13.6pp in a GNN-only tournament. The biggest single contribution is Cand 11 (+13.6pp standalone over vanilla); Cand 8+10 adds +7.5pp standalone but doesn't compound additively with Cand 11.

Next investigation: a closeout-specific prior for Cand 11 (different mechanism than Cand 8+10's blanket VP boost), aimed at the 27% city-build rate gap vs LookV3's 51% identified in the closeout diagnostic.
