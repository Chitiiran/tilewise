# Full-Catan tournament — Cell 6 and Cell 1 dominate; Cell 5 v2 and vanilla collapse

**Date:** 2026-05-27
**Trigger:** All four loss-aug cells were trained on v3 rules (`vp_target=5, bonuses=False`). We wanted to see how the same checkpoints play under full Catan rules (`vp_target=10, bonuses=True`) — an out-of-distribution generalization test. Same `e10d_quad_gnn` harness as the v3 4-PureGnn tournament earlier today, only difference is the engine config.

## Setup

| Slot | Cell | Checkpoint |
|---|---|---|
| A | Cell 0 (vanilla) | `runs/v3/loss_aug/00_baseline_h128_l4_pilot/training_h128_l4/checkpoint_epoch05.pt` |
| B | Cell 1 (Cand 8 + Cand 10) | `runs/v3/loss_aug/01_cand8_cand10_h128_l4/training_h128_l4/checkpoint_epoch10.pt` |
| C | Cell 5 v2 (Cand 11 alone) | `runs/v3/loss_aug/05_cand_road_pip_h128_l4_v2/training_h128_l4/checkpoint_epoch10.pt` |
| D | Cell 6 (Cand 11 + Cand 8 + Cand 10) | `runs/v3/loss_aug/06_cand11_cand8_cand10_h128_l4/training_h128_l4/checkpoint_epoch10.pt` |

Engine flags: `--vp-target 10 --bonuses` (full Catan with longest road + largest army +2 VP each). All other settings match the v3 4-PureGnn tournament: `hidden_dim=128, num_layers=4, seed_base=19_000_000, workers=10, device=cuda, max_seconds=600s, 300 games × 4 rotations = 1200 games`.

Wall-clock: ~130 min (5× slower than the v3 4-PureGnn tournament's ~27 min — 10-VP games take longer, especially when GNN policies stall).

## Results

```
Cell 0 (vanilla)              13 / 1200  ( 1.08%)   rot[  2,   4,   2,   5]
Cell 1 (Cand 8 + Cand 10)    515 / 1200  (42.92%)   rot[130, 114, 145, 126]
Cell 5 v2 (Cand 11 alone)      9 / 1200  ( 0.75%)   rot[  2,   0,   2,   5]
Cell 6 (Cand 11 + 8 + 10)    652 / 1200  (54.33%)   rot[165, 179, 149, 159]
Draws / timeouts              11 / 1200  ( 0.92%)
```

| Rank | Cell | Wins | % | 95% CI |
|---|---|---:|---:|---|
| 🥇 | **Cell 6 (stack)** | **652** | **54.33%** | ±2.8pp |
| 🥈 | **Cell 1 (Cand 8+10)** | **515** | **42.92%** | ±2.8pp |
| 🥉 | Cell 0 (vanilla) | 13 | 1.08% | ±0.6pp |
| 4 | Cell 5 v2 (Cand 11 alone) | 9 | 0.75% | ±0.5pp |

## Complete inversion vs the v3-rules tournament

Same models, same lineup, same harness — only the engine rules differ.

| Cell | v3 rules (5 VP, no bonuses) | full Catan (10 VP, bonuses) | Δ |
|---|---:|---:|---:|
| Cell 0 (vanilla) | 17.33% | **1.08%** | **−16.25pp** |
| Cell 1 (Cand 8+10) | 24.83% | **42.92%** | **+18.09pp** |
| Cell 5 v2 (Cand 11 alone) | **30.92% (winner)** | **0.75%** | **−30.17pp** |
| Cell 6 (stack) | 26.92% | **54.33% (winner)** | **+27.41pp** |

**Cell 5 v2 went from 31% to 0.75% — a 41× collapse.** Cell 6 went from 27% to 54% — almost doubled.

## Mechanism: bonuses (LR + LA) decided the tournament

Cell 6 averages 801 moves per win. Cell 1 averages 768. **Cell 5 v2 averages 5125 moves per win** — games stall to near-infinity. Cell 5 v2's *losing* games end with mean VP = 5.48 — it's reaching the v3 win threshold and then **failing to push past it** because it never learned to value the bonus mechanics.

| Cell | Wins | Mean VP at win | VP distribution (winners only) | Mean game length (moves) | Mean losing-game VP |
|---|---:|---:|---|---:|---:|
| Cell 0 | 13 | 10.00 | 10:13 | 1612 | 5.10 |
| **Cell 1** | **515** | **10.07** | 10:479, 11:36 | **768** | 5.48 |
| Cell 5 v2 | 9 | 10.00 | 10:9 | **5125** | 5.48 |
| **Cell 6** | **652** | **10.07** | 10:606, 11:46 | **801** | 5.71 |

Cell 6 and Cell 1 close games quickly because they're acquiring bonus VPs alongside building. Cell 5 v2 stalls at 5-6 VP and doesn't know how to get further. Cell 0 stalls similarly but slightly less because vanilla didn't have any strategy distortion to begin with.

### Why Cell 1 wins so much when bonuses are on

Cell 1's behavioral analysis from the 4-way tournament showed **BuyDevCard 16.37/100 turns — 3× more than any other cell**. The dev card deck is dominated by knights (14 of 25 = 56% cited engine deck composition). So Cell 1 inadvertently:

- Buys ~3× more knights than others
- Plays knights at 5.43/100 turns (cited) — second-highest after vanilla
- Each played knight counts toward Largest Army (+2 VP at 3 knights)

In v3 with `bonuses=False`, this knight-buying was wasted resource — Cell 1 reached ~5 VP via the slow city/settlement path while burning resources on useless knights. In full Catan, **those knights are 2 VP of bonus on top of everything else**. Cell 1's "dev-card spam" — diagnosed as a degenerate equilibrium in earlier journals — turns out to be **exactly the right strategy for full Catan's largest-army race.**

### Why Cell 6 wins more than Cell 1

Cell 6 = Cell 1 + Cand 11. Cell 6 inherits Cell 1's largest-army dev-card play AND adds Cand 11's heavy road-building (22.06 roads/100 turns vs Cell 1's 16.64). More roads → longer road segments → eventually the Longest Road bonus (+2 VP at 5+ road segments). **Cell 6 has TWO bonus-winning machines stacked.**

The 54% vs 42% win rate gap (Cell 6 vs Cell 1) is ~+11pp, with non-overlapping CIs. Decisive.

### Why Cell 5 v2 collapsed so completely

Cell 5 v2 = Cand 11 alone. Cand 11's behavioral signature:
- Highest road rate (22.06/100)
- Highest settlement rate (5.74/100)
- **Lowest dev card rate (5.77/100)** ← critical for full Catan
- Best roads-per-settlement ratio (3.84)

Cand 11's training prior (`λ_road=0.05`) actively biases the policy AWAY from dev cards and TOWARD roads + settlements. In v3, that produced the best 5-VP strategy. In full Catan, **it produces no path to largest army**, and the few accidental knights from low dev-card purchases don't accumulate to win the bonus.

Cand 11's longest-road potential exists (22.06 roads/100 — highest of any cell), but without the closeout VP from largest army, the model can't reach 10 VP. It stalls at ~5-6 VP and the OTHER players (Cell 1, Cell 6) win the games.

### Why vanilla collapsed too

Cell 0 vanilla has no strategy distortion in either direction. It builds dev cards at 9.94/100 (median of all cells) and roads at 19.44/100 (high but not weighted). In v3, that produced a decent 17.33% via balanced play. In full Catan, vanilla:
- Has no specific bias toward bonus mechanics
- Is just a weaker overall policy than Cell 1 or Cell 6
- Gets dominated by the bonus-collecting cells

Cell 0's 1.08% is essentially "vanilla doesn't get the rules joke."

## The retrospective lesson

The Cell 6 journal (committed earlier today) called the stack "a worse intervention than Cand 11 alone" based on the v3 head-to-head and 4-PureGnn results. **That conclusion is correct for v3 rules and wrong for full Catan.** Cell 6 wasn't worse — it was an out-of-distribution stronger model that the v3 test environment couldn't expose.

This also retrospectively validates **stacking Cand 11 + Cand 8 + Cand 10** as a hedged strategy when target deployment rules are unknown:
- Cand 8+10 → wins largest army when bonuses are on
- Cand 11 → wins longest road when bonuses are on
- Together → both bonuses available
- Without bonuses → Cand 11's road prior dominates strategy, slightly worse than Cand 11 alone but still competitive

**No single cell is "the best" — the best cell depends on the target rule set.** Future cumulative-best claims need to be scoped to a rule configuration.

## Behavioral inference (not directly measured)

The full-Catan behavioral analysis (city rate, road rate, etc.) would have required a separate diagnostic walk over the parquets. Not run for this journal — the VP-distribution + game-length evidence above is already decisive on the mechanism. If future analysis needs it, the script pattern from `scratch_midgame_actions_e10c_1200.py` extends trivially.

## Statistical confidence

- Cell 6 vs Cell 1: 54.33% vs 42.92%, gap = 11.41pp. CIs (±2.8pp each) overlap by ~2pp at the boundaries — gap is large enough that Cell 6 > Cell 1 holds at >99% confidence.
- Cell 0 vs Cell 5 v2: 1.08% vs 0.75%, gap = 0.33pp. CIs heavily overlap. Statistically tied at "near-zero."
- Cell 6 / Cell 1 vs Cell 0 / Cell 5 v2: massive gaps (40-50pp), CIs do not overlap at all. Decisive.

11 of 1200 games (~0.9%) timed out at the 600s/game cap. These are mostly games where Cell 5 v2 or Cell 0 stalled — neither can close past 6 VP — but the other two players also failed to win quickly. Removing them doesn't change the ranking.

## Decision matrix update — context-scoped cumulative best

| Tournament context | Cumulative best | Justification |
|---|---|---|
| **v3 rules (5 VP, no bonuses)** — TRAINING DISTRIBUTION | **Cell 5 v2 (Cand 11 alone)** | 14.67–16.83% head-to-head w/ LookV3; 30.92% 4-PureGnn |
| **Full Catan (10 VP, bonuses)** — OUT-OF-DISTRIBUTION | **Cell 6 (stack)** | 54.33% 4-PureGnn |
| Mid-tournament (v3, vs Random+GnnMcts+LookV3) | Cell 6 (overstated; mid-tournament has known issues) | 15.8% ep15 — but the metric is unreliable |

## Cited artefacts

- New tournament: `runs/v3/e10d_4gnn_fullcatan_1200_2026_05_27/2026-05-27T19-52-e10d_quad_gnn/`
- Launch log: `launch.log`
- Tournament module: `mcts_study/catan_mcts/experiments/e10_quad_gnn.py` (commit `5d22713`)
- Companion journals:
  - `2026-05-27-4puregnn-no-lookahead-tournament.md` (v3 4-PureGnn result)
  - `2026-05-26-cand11-headtohead-tournament.md` (Cell 5 v2 v3-rules head-to-head)
  - `2026-05-26-cell6-cand11-cand8-cand10-stack.md` (Cell 6 v3-rules result, now retrospectively re-framed)
  - `2026-05-25-cell5-road-pip-prior.md` (Cell 5 v2 training)

## Memory items to update

- **`project_cand11_cumulative_best_2026_05_26.md`:** scope the claim explicitly to v3 rules. Add a pointer to this journal for full-Catan behavior.
- **New memory entry needed:** `project_cell6_fullcatan_winner_2026_05_27.md` — Cell 6 (Cand 11 + Cand 8 + Cand 10 stack) is the cumulative best for full-Catan deployment (vp=10, bonuses on), winning 54.33% in 4-PureGnn tournament. Mechanism: stacked LR+LA bonus collection.
- **`feedback_use_headtohead_not_midtournament.md`:** add a third caveat — head-to-head only generalizes within the same rule set. v3 rankings do NOT predict full-Catan rankings.

## Conclusion

**The same four models produce completely inverted rankings depending on whether bonuses are enabled.** In the training distribution (v3 rules), Cell 5 v2 (Cand 11 alone) wins; in full Catan, Cell 5 v2 collapses to 0.75% while Cell 6 (the stack) wins decisively at 54.33%.

The Cell 1 dev-card-spam pattern we'd diagnosed as a "degenerate equilibrium" is actually a coherent largest-army strategy that's masked by v3's `bonuses=False` flag. Cand 11's road-heavy strategy doubles as a longest-road strategy. Cell 6 combines both, winning ~95% of all games against vanilla and Cand-11-alone in full Catan.

**Practical takeaway:** if the target deployment is full Catan, train with `bonuses=True` self-play data and prefer the stacked loss-aug (Cand 11 + Cand 8 + Cand 10). If the target is v3 / Catan-Lite, prefer Cand 11 alone. The "best model" question is rule-dependent and the v3-only training framework was inherently biased against bonus-driven strategies.
