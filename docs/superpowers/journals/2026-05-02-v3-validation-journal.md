# v3.4 Journal — Validation sweep (1k games + 5-epoch GNN + e10 tournament)

**Date:** 2026-05-02
**Branch:** `v3` @ post-39cbdb2
**Tasks:** v3.4 (validation sweep)
**Predecessor journal:** [v3 phase 1](./2026-05-02-v3-phase1-journal.md)

## TL;DR

The v3 pipeline runs end-to-end. Data generation took 8:20 for 1k games (under
the 15-min spec target). Training reached val_top1=0.199 in 5 epochs. **The
trained GNN won 0/20 seats against LookaheadMctsV3 in the e10 tournament** —
same shape as the v2 Phase 3 outcome. The validation run is not enough data
to beat the lookahead baseline; v3.5 (production sweep) will need a much
larger dataset.

The infrastructure is sound; the data scale isn't.

## What ran

### v3 data generation (e9)

```
python -m catan_mcts run e9 --out-root runs/v3 \
  --num-games 1000 --base-sims 200 --lookahead-depth 10 \
  --workers 8 --max-seconds 300
```

**Output:** `runs/v3/2026-05-02T12-08-e9_v3_data_gen/`

| Metric | Value |
|---|---|
| Games | 1000 |
| Recorded MCTS moves | 26,267 (~26 per game) |
| Wall-clock | 8m 20s (8 workers) |
| Per-game (effective) | 0.5s (parallel) / ~4s (serial) |
| Game length | median 198, mean 207, max 470 steps |
| Max VP at terminal | 5 in all 1000 games (rule check ✓) |
| Win distribution | seat 0: 249, 1: 255, 2: 265, 3: 231 (balanced) |
| Recorded-player rotation | 6732 / 6663 / 6575 / 6297 (balanced) |

### Training

```
python -m catan_gnn.train --run-dirs runs/v3/2026-05-02T12-08-e9_v3_data_gen \
  --out-dir runs/v3/training_validation --epochs 5 --device auto \
  --rotate --rotate-mode random --cache-path runs/v3/cache_validation.pt
```

**Output:** `runs/v3/training_validation/checkpoint_best.pt` (epoch 2)

| Epoch | train_loss | val_loss | val_top1 |
|---|---|---|---|
| 1 | 2.822 | 2.784 | 0.159 |
| **2** | **2.772** | **2.808** | **0.199** ← best |
| 3 | 2.747 | 2.847 | 0.172 |
| 4 | 2.720 | 2.854 | 0.176 |
| 5 | 2.692 | 2.858 | 0.177 |

Wall-clock: ~3.5 min for 5 epochs on GTX 1650, including a 52-second cache
build and 26k positions. `--rotate-mode random` enabled (each batch sees a
mix of 0..5 hex-rotations).

**Reading the curves.** train_loss decreases monotonically, val_loss increases
after epoch 2 — the canonical overfit signature with this small dataset.
val_top1 plateaus at 0.17–0.20. For comparison the v2 baseline GNN on its
larger v2 dataset hit val_top1=0.27 at epoch 4. v3's 0.20 with 1k games (vs
v2's much larger dataset) is **not bad per-position** — the simpler reward
landscape really does carry more signal per move — but the absolute number
of positions is too small for the model to beat lookahead.

### e10 tournament

```
python -m catan_mcts run e10 --out-root runs/v3 \
  --checkpoint runs/v3/training_validation/checkpoint_best.pt \
  --num-games-per-seating 5 --sims 100 --lookahead-depth 10 --base-sims-v3 200 \
  --workers 4 --device auto
```

**Output:** `runs/v3/2026-05-02T12-24-e10_v3_tournament/`

20 games (5 per cyclic rotation × 4 rotations).

| Player | Wins | % |
|---|---|---|
| LookaheadMctsV3 | 20 | 100% |
| GnnMcts | 0 | 0% |
| PureGnn | 0 | 0% |
| Random | 0 | 0% |

LookaheadMctsV3 won every rotation in every seat. All 20 games terminated
correctly at exactly 5 VP, median game length 177 moves.

## Verdict on spec §12 success criterion #3

> "After ≥5 epochs of training on a v3 dataset, a 4-way `e9` tournament gives
> `PureGnn` or `GnnMcts` ≥1/4 seat wins (currently 0/4 in v2)."

**FAIL** at the 1k-game validation scale. Same outcome as v2 Phase 3.

This is the expected outcome of a *validation* run — the spec called it
out as "1k games to prove the pipeline." The pipeline works (e9 → train →
e10 ran cleanly with no crashes, all recorded games terminated, all parquet
schemas matched, GPU training came up clean). The question of *whether v3
ever produces a GNN that beats lookahead* needs the production sweep.

## Implications for v3.5 (production sweep)

**Data scale.** v2's writeup-quality GNN training used ~120k positions across
multiple sweeps. v3's 26k positions clearly aren't enough. Targets for the
production sweep:
- **5-10× more games** (5-10k v3 games) → 130-260k positions before
  rotation augmentation, 780k-1.5M with random rotation. Closer to the v2
  data scale but with the v3 reward signal.
- **Per-game wall-clock budget:** at 4-5s per game with 8 workers, 5k games
  is ~50 min, 10k games is ~100 min. Both fit a single overnight session.
- **Training budget:** 20-30 epochs (let val_top1 climb past the plateau).

**What stays the same:**
- Engine flags (`vp_target=5, bonuses=False`).
- LookaheadMctsV3 schedule (max(50, 200 * 0.7^vp)).
- Recorded-player rotation per seed.
- Random rotation augmentation during training.

**What might change:**
- Increase `lookahead_depth` from 10 → 25 (stronger labels, slower games).
- Increase `base_sims` from 200 → 400 at acting_vp=0 (cleaner policy
  targets at game start where decisions matter most).
- Stay at `lookahead_depth=10, base_sims=200` if production timing
  tightens — wall-clock is more important than label quality at this stage.

## What's next

- Mark v3.4 complete.
- v3.5 (production sweep): scale up data gen to 5-10k games, train to
  convergence, run a fuller e10 tournament (e.g. 50 games per rotation =
  200 total) to detect partial wins (1/200 = 0.5% is detectable; 1/20 was
  not granular enough).

## Time spent

- e9 data gen: 8m 20s (background)
- training: 3m 30s
- e10 tournament: ~10 min
- analysis + writeup: ~10 min

**Total: ~32 min of actual compute** for the validation run. A 10k-game
production sweep at the same per-game rate would be ~85 min; a 50-rotation
tournament would be ~3 hours. Both fit in a single overnight session.
