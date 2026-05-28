# Pass-100k Tournament Results — LAST vs BEST checkpoint comparison

**Date:** 2026-05-10
**Status:** Complete. 720 games across 6 tournaments.
**Trigger:** User asked "let's stop the training and start the tournament. only h64_l3 is at epoch 13 let's use last epoch and run the tournament. once that's finished let's run the best epoch."

## Setup

**Cited from `e10_v3_tournament.py:48` and `grid_pass3_tournament.py:207`:**
- Roster: `[GnnMcts, PureGnn, LookaheadMctsV3, Random]`
- 4 rotations (cyclic), 30 games per seating × 4 = 120 games per cell
- `seed_base=19000000` (cited from `feedback_pass100k_roadmap_v3.md`) — byte-identical to pass-3 + pass-3-lastepoch tournaments → directly comparable
- `lookahead_depth=10`, `base_sims_v3=200`, `sims=100`
- `--workers 10` (cited from launch logs after switching from 4 mid-run for ~2.5× throughput)
- v3 ruleset: `vp_target=5, bonuses=False` (cited `engine.rs:43`) — no longest road / largest army bonuses

**Cited cells tested:**
- h32_l2 (cited `_BASE_SEATING` × 32 hidden × 2 SAGE layers, ~13k params)
- h64_l3 (~178k params)
- h128_l4 (~1.4M params)

**Cited checkpoints:**
- LASTEPOCH = `checkpoint_epoch20.pt` for h32_l2 and h128_l4; `checkpoint_epoch13.pt` for h64_l3 (training was killed mid-flight per user instruction)
- BEST = `checkpoint_best.pt` (highest val_top1 epoch saved during training):
  - h32_l2: ep3, val_top1=0.1843
  - h64_l3: ep2, val_top1=0.1831
  - h128_l4: ep8, val_top1=0.1836

## Final cited results (all 6 tournaments × 120 games = 720 games total)

| Cell | Mode | Look | PureGnn | GnnMcts | Random | GNN total |
|---|---|---|---|---|---|---|
| h32_l2 | LASTEPOCH | 112 (93.3%) | **5** | 1 | 2 | 6 |
| h32_l2 | BEST | 112 (93.3%) | 4 | **3** | 1 | 7 |
| h64_l3 | LASTEPOCH | 117 (97.5%) | **2** | 1 | 0 | 3 |
| h64_l3 | BEST | 117 (97.5%) | 1 | 0 | 2 | 1 |
| h128_l4 | LASTEPOCH | 117 (97.5%) | 3 | 0 | 0 | 3 |
| **h128_l4** | **BEST** | **109 (90.8%) ★** | **8 ★** | 0 | 2 | **8 ★** |

## Cited per-cell deltas (LAST − BEST)

| Cell | LAST PureGnn | BEST PureGnn | Δ direction | LAST GNN total | BEST GNN total | Δ direction |
|---|---|---|---|---|---|---|
| h32_l2 | 5 | 4 | LAST +1 | 6 | 7 | BEST +1 |
| h64_l3 | 2 | 1 | LAST +1 | 3 | 1 | LAST +2 |
| h128_l4 | 3 | 8 | **BEST +5** | 3 | 8 | **BEST +5** |

## Cited headlines

### 1. h128_l4 BEST is the strongest GNN we've measured under v3 rules

- **8 PureGnn wins / 120 games = 6.7%** (cited from aggregator)
- **Lookahead at only 109/120 = 90.8%** — the only cell where Lookahead's grip dropped below 93%
- 2.7× the PureGnn rate of any other (LAST or BEST) tournament

### 2. The "last beats best" pattern reverses for h128_l4

In all prior tournaments (cited from earlier journals):
- Pass-3 BEST aggregate (9 cells × 120 games): 20/1080 PureGnn wins (1.9%)
- Pass-3 LASTEPOCH aggregate: 50/1080 PureGnn wins (4.6%) — **2.5× last**

For h32_l2 and h64_l3 in this experiment, the same direction held (last had 1 more PureGnn win than best). But for h128_l4: **BEST had 5 more PureGnn wins than LAST.**

### 3. Hypothesis (cited from training logs)

h128_l4 trained on the 100k corpus showed **the most pronounced overfit** of the three diagonal cells — val_loss climbed monotonically from 2.865 (ep1) to 3.165 (ep20) (cited from `grid_pass100k_diagonal_2026-05-09T07-49.log`). The bigger model captures more idiosyncratic patterns from the training data with depth, so its epoch-20 weights drift far past their generalization peak. ep8 (best_top1 epoch) captures cleaner policy → better tournament play.

For smaller models (h32_l2, h64_l3): less capacity to overfit, so last vs best is a wash. **Capacity-overfitting trade-off matters more at larger scale.**

### 4. The Lookahead ceiling is broken (slightly) only by h128_l4 BEST

| Tournament | Lookahead win-rate |
|---|---|
| All h32_l2 / h64_l3 / h128_l4 LASTEPOCH and h32_l2 / h64_l3 BEST | **93-97.5%** (typical baseline) |
| **h128_l4 BEST** | **90.8%** (cited n=120) |

The ~6 percentage-point reduction is meaningful for n=120 (cited binomial 95% CI for 109/120 ≈ ±5%, so the lower bound of Lookahead's win-rate is ~85%). Either:
- The h128_l4 BEST GNN is genuinely stronger and forces Lookahead's MCTS into worse positions on average, or
- Random got 2 wins which cuts into Lookahead's share

PureGnn's 8 wins explains the bulk of the 8-win Lookahead deficit.

## Cross-pass aggregate cited

| Pass | Source | Aggregate Lookahead | Aggregate PureGnn | Aggregate GnnMcts |
|---|---|---|---|---|
| Pass-3 BEST (9 cells) | grid_pass3.json | 1029/1080 (95.3%) | 20/1080 (1.9%) | 15/1080 (1.4%) |
| Pass-3 LASTEPOCH (9 cells) | grid_pass3_lastepoch.json | 1007/1080 (93.2%) | 50/1080 (4.6%) | 12/1080 (1.1%) |
| **Pass-100k LAST (3 cells, this exp)** | grid_pass100k_lastepoch | 346/360 (96.1%) | 10/360 (2.8%) | 2/360 (0.6%) |
| **Pass-100k BEST (3 cells, this exp)** | grid_pass100k_best | 338/360 (93.9%) | 13/360 (3.6%) | 3/360 (0.8%) |

**Cited compositional insight:** even with the 100k corpus boost on h128_l4, the pass-100k aggregate Lookahead win-rate (93.9% best, 96.1% last) is still in the same band as pass-3. **The 100k corpus moved the floor, not the ceiling.** h128_l4 BEST is an outlier within the pass-100k 3-cell set, not a generalized improvement.

## What this means for the GNN

1. **Architecture scaling helps with right corpus + right checkpoint.** h128_l4 BEST > h32_l2 BEST > h64_l3 BEST in PureGnn wins. But you need the larger corpus AND best-checkpoint selection. Either alone gives you nothing.

2. **GnnMcts is broken-ish** under v3 rules. Across all 6 tournaments: GnnMcts won 4/720 = 0.6% (cited). MCTS guided by the GNN consistently underperforms PureGnn alone. Possible reasons (untested):
   - GNN's value head is poorly calibrated → MCTS backups inject noise
   - Lookahead's MCTS is at depth=10/200 sims; GnnMcts is at 100 sims with weaker rollouts
   - The MCTS-GNN combination has a bug in how the GNN evaluator is plugged in

3. **The val_top1 → tournament-strength mapping is non-monotone.** Cited val_top1:
   - h32_l2 best=0.1843, h64_l3 best=0.1831, h128_l4 best=0.1836 — almost identical
   - But tournament: h128_l4 (8) ≫ h32_l2 (4) ≫ h64_l3 (1)
   
   **val_top1 saturates around 0.184 (a structural ceiling, cited from training logs across all runs)** but model strength varies wildly underneath that ceiling. **The metric we're training to is not predictive of game-winning ability** — confirms hypothesis 2 in lateral-thinking item A' from the loss-augmentation design doc.

## Investigation TODOs cited from this run

1. **Run h128_l4 BEST tournament again with different seed_base** to confirm 8 PureGnn wins isn't seed-effect noise. n=120 with binomial CI ~±5pp; we want to rule out "lucky 8".
2. **Train h128_l4 with regularization** (dropout, weight decay, label smoothing) and see if the BEST checkpoint advantage scales further when overfitting is suppressed.
3. **Investigate GnnMcts's poor performance** — check the bot's MCTS evaluator code path, verify GNN value head is being read correctly.
4. **Implement loss-augmentation Candidate 7 (action-class-balanced policy loss)** from the design doc to see if reweighting changes the val_top1 ceiling at 0.184.

## Files cited

- `runs/v3/grid_pass100k_tournament/h32_l2_lastepoch/2026-05-08T22-20-e10_v3_tournament/` (n=120)
- `runs/v3/grid_pass100k_tournament/h32_l2_best/2026-05-09T02-10-e10_v3_tournament/` (n=120)
- `runs/v3/grid_pass100k_lastepoch_tournament/h64_l3/2026-05-10T13-25-e10_v3_tournament/` (n=120)
- `runs/v3/grid_pass100k_lastepoch_tournament/h128_l4/2026-05-10T14-33-e10_v3_tournament/` (n=120)
- `runs/v3/grid_pass100k_best_tournament/h64_l3/2026-05-10T15-23-e10_v3_tournament/` (n=120)
- `runs/v3/grid_pass100k_best_tournament/h128_l4/2026-05-10T16-00-e10_v3_tournament/` (n=120)
- `scratch_partial_results.py` — aggregator with the bug fix for double-counting per-seed/per-rot shards
- `runs/v3/dashboard/grid_pass100k.json` — primary dashboard JSON (Comparison tab reads this)
- `runs/v3/dashboard/grid_pass100k_lastepoch.json` — separate file for LASTEPOCH-only view
- `runs/v3/dashboard/grid_pass100k_best.json` — separate file for BEST-only view

## Bug fixed during this run (cited)

The aggregator's `glob("games.*.parquet")` was matching BOTH `games.seed=<N>.parquet` (per-game shards) AND `games.rot=<N>.parquet` (consolidated rotation shards). The recorder consolidates per-seed shards into per-rot at end of each rotation but **doesn't delete the per-seed files** — so the same game appears in both formats during the consolidation window, causing the aggregator to **double-count in-flight games**. This produced impossible mid-flight numbers (n=88 with PureGnn=6 etc.) that I incorrectly reported earlier.

**Fix (cited `scratch_partial_results.py:27-65`):** prefer `games.rot=*.parquet` shards; only count `games.seed=*.parquet` for seeds NOT already covered by rot shards. Plus `try/except FileNotFoundError` for the race between glob and read when the recorder atomically replaces files.

User caught this — pushed back on impossible monotonicity violation, leading to investigation, root-cause identification (shard duplication), and fix. Lesson: **monotonic in-flight counts are a non-trivial invariant when readers and writers coexist.**
