# Cell 6: Cand 11 + Cand 8 + Cand 10 stacked

**Date:** 2026-05-26
**Status:** COMPLETED — trained 15 epochs cleanly; 1200-game head-to-head shows the stack does NOT beat Cell 5 v2 (Cand 11 alone). Mid-tournament numbers (15.0% / 15.8%) substantially overstated the stack's strength.
**Hypothesis:** Cand 11 fixes road-positioning (Cell 5 v2's strength: highest road, settle, ending-pip among GNNs) and Cand 8 fixes city closeout (Cell 1's strength: cities/turns-with-resources = 0.656, highest of any role). Stacking should compound: Cand 11 reaches city-ready state often, Cand 8 closes when there.

**Plan:** Direct stack of the three flags as they already exist. No new code. If the stack works (head-to-head > 16.83%), it's the new cumulative best with zero design risk. If it shows the Cell-1-style dev-card spam regression, fine-tune `λ_vp` lower in Cell 7.

## Setup

| Setting | Cell 6 | Comparison |
|---|---|---|
| Architecture | h128_l4 (632k params) | same as Cell 5 v2 |
| Cache | cache_100k.pt | same |
| Batch size | 256 | same |
| LR | 1e-3 Adam | same |
| Augmentation | random hex rotation | same |
| Seed | 0 | same |
| `lambda_road` (Cand 11) | **0.05** | matches Cell 5 v2 |
| `lambda_vp` (Cand 8) | **0.10** | matches Cell 1 |
| `vp_compare_rule` (Cand 10) | **True** | matches Cell 1 |
| Mid-tournaments | every 5 epochs, 120 games, seed_base=19M | same |

## Diagnostic backing the design

From the 1,200-game e10c head-to-head (cited `2026-05-26-cand11-headtohead-tournament.md`) + closeout diagnostic (cited `scratch_cand11_closeout_diagnostic.py`):

| Role | turns_with_city_resources | cities_built / turns_with_city_resources |
|---|---:|---:|
| Cell 5 v2 (Cand 11) | **3019** (highest) | **0.271** (low) |
| Cell 1 (Cand 8+10) | 279 (low) | **0.656** (highest) |
| LookaheadV3 | 2334 | 0.510 |

Cand 11 reaches the city-ready state 11× more often than Cell 1; Cell 1 converts those moments 2.4× better. Independently bottlenecked. Stacking should yield Cand 11's positioning × Cand 8+10's closeout efficiency.

## Pre-launch checks

- ✅ Smoke test `test_cell6_smoke.py` — stack runs 1 epoch without NaN, all-flags-off byte-identical to vanilla. Commit `f905930`.
- ✅ No concurrent CPU-heavy processes.
- ✅ Memory 53 GB free pre-launch.
- ⏸ Calibration (gate-firing rates) — relying on the individual Cand 11 and Cand 8 calibrations from their respective cells; no new mechanism added.

## Launch command (PID 4702)

```bash
python scripts/train_grid_inproc.py \
  --cache-path ~/catan_cache/cache_100k.pt \
  --out-root runs/v3/training/loss_aug/06_cand11_cand8_cand10_h128_l4 \
  --status-file runs/v3/dashboard/cell6.json \
  --epochs 15 --batch-size 256 --device auto \
  --rotate --rotate-mode random --cells h128_l4 --seed 0 \
  --mid-tournament-every 5 \
  --lambda-road 0.05 \
  --lambda-vp 0.10 \
  --vp-compare-rule
```

## Expected timeline (per Cell 5 v2 reference)

- Cache load: ~40 min
- ep1: ~58 min (with vectorized Cand 11)
- ep5: ~5h post-launch → mid-tournament ~6h
- ep10 mid-tournament: ~12h
- ep15 + final mid-tournament: ~19h
- Then 1200-game head-to-head: +~67 min

Total to result: ~20h.

## Decision rules

- **Early kill (ep5):** if PureGnn ≤ 12/120 (≤10%), kill per the `mid_tournament_drop_threshold` rule + plan §10.3.
- **Behavioral regression check (ep5 parquets):** if BuyDevCard/100 > 14 (close to Cell 1's 16.37), flag dev-card spam regression. May warrant kill even if winrate is decent.
- **Cumulative-best (ep10 head-to-head):** must beat Cell 5 v2's 16.83% at n=1200 with non-overlapping CI to claim cumulative-best.

## Per-epoch metrics

| ep | train_loss | val_loss | val_top1 | vp_swap | best? |
|---:|---:|---:|---:|---:|---|
| 1 | 2.838 | 3.231 | 0.178 | 18.68% | ✓ |
| 2 | 2.799 | 3.293 | 0.170 | 18.69% | — |
| 3 | 2.733 | 3.332 | 0.183 | 18.69% | ✓ new best |
| 4 | 2.665 | 3.453 | 0.172 | 18.69% | — |
| 5 | 2.616 | 3.429 | 0.176 | 18.69% | — |
| 6 | 2.581 | 3.478 | 0.180 | 18.70% | — |
| 7 | 2.554 | 3.519 | 0.176 | 18.70% | — |
| 8 | 2.532 | 3.523 | 0.166 | 18.70% | — |
| 9 | 2.515 | 3.552 | 0.173 | 18.70% | — |
| 10 | 2.501 | 3.548 | 0.174 | 18.70% | — |
| 11 | 2.490 | 3.522 | 0.173 | 18.70% | — |
| 12 | 2.480 | 3.577 | 0.175 | 18.70% | — |
| 13 | 2.471 | 3.582 | 0.174 | 18.70% | — |
| 14 | 2.464 | 3.592 | 0.167 | 18.70% | — |
| 15 | 2.457 | 3.579 | 0.168 | 18.71% | — |

train_loss monotone decreasing 2.838→2.457; val_loss climbing steadily
2.865→3.579 (Cell 1's overfit signature). val_top1 oscillating 0.166–0.183.
vp_swap rate locked at ~18.7% (Cand 10's swap behavior byte-identical to
Cell 1's at every epoch — Cand 11 doesn't interfere). Best val_top1 at
ep3; `checkpoint_best.pt` points there but per the standing rule it's
val-best, not tournament-best.

Wall-clock: ~19h training (60.3 min ep1, slight CPU contention pushed
average to ~65 min/epoch) + 3 × 45 min mid-tournaments.

## Mid-tournament results

120 games per mid-tournament (30 × 4 rotations), sims=100,
lookahead_depth=10, base_sims_v3=200, h128_l4, workers=10, device=cuda,
seed_base=19M. STANDARD_* config from `mid_training_tournament.py`.

| ep | PureGnn | GnnMcts | LookV3 | Random | draws | elapsed |
|---:|---:|---:|---:|---:|---:|---:|
| 5  | 13 (10.8%) | 1 (0.8%) | 105 (87.5%) | 1 | 0 | 2726s |
| 10 | 18 (15.0%) | 1 (0.8%) | 101 (84.2%) | 0 | 0 | 2893s |
| 15 | 19 (15.8%) | 1 (0.8%) |  98 (81.7%) | 1 | 1 | 2681s |

**Monotone-rising mid-tournament trajectory** — first time in the
loss-aug roadmap a cell improved at every checkpoint. No early-kill
trigger (≤12/120 at ep5). No `mid_tournament_drop_threshold` trigger
(largest drop allowed = 3 games; Cell 6 only went up).

**LookV3 share dropped 105 → 101 → 98** — the lowest LookV3 share in any
mid-tournament across all cells. Looked like Cell 6 was beating Lookahead
more than any prior cell.

## Comparison with Cell 5 v2 mid-tournaments

| Cell | ep5 | ep10 | ep15 |
|---|---:|---:|---:|
| Cell 0 vanilla | 12.5% | 1.67% | — |
| Cell 1 (Cand 8+10) | 10.83% | 9.17% | 10.00% |
| Cell 5 v2 (Cand 11 alone) | 5.8% | 10.8% | 5.8% (U-shape) |
| **Cell 6 (stack)** | **10.8%** | **15.0%** | **15.8%** (monotone) |

Looked compelling. Then we ran the head-to-head.

## 1200-game head-to-head — THE REAL TEST

Lineup: Cell 6 ep10 + Cell 5 v2 ep10 + Cell 1 ep10 + LookV3. Same e10c
config as the 2026-05-26 head-to-head against Cell 0 vanilla. Wall-clock
~25 min (Cell 6 in slot A, Cell 5 v2 in slot B, Cell 1 in slot C).

```
Cell6-stack-ep10         107 / 1200  ( 8.92%)   rot[27, 30, 27, 23]
Cell5v2-cand11-ep10      176 / 1200  (14.67%)   rot[40, 45, 43, 48]
Cell1-cand8cand10-ep10   106 / 1200  ( 8.83%)   rot[34, 32, 25, 15]
LookaheadV3              811 / 1200  (67.58%)   rot[199, 193, 205, 214]
Draws                      0 / 1200
```

| Player | Wins / 1200 | % | 95% CI | vs cumulative best (Cell 5 v2 16.83%) |
|---|---:|---:|---|---|
| LookaheadV3 | 811 | 67.58% | ±2.7pp | — |
| **Cell 5 v2 (Cand 11)** | **176** | **14.67%** | ±2.0pp | within noise of 16.83% (previous run) |
| **Cell 6 (stack)** | **107** | **8.92%** | ±1.6pp | **−7.91pp** from Cell 5 v2 here |
| Cell 1 (Cand 8+10) | 106 | 8.83% | ±1.6pp | — |

**Cell 6 is statistically tied with Cell 1** (8.92% vs 8.83%, gap 0.09pp,
CIs heavily overlap). The stack added nothing over Cand 8+10 alone in
head-to-head, despite the mid-tournament number being almost 2× higher.

**Cell 5 v2 (Cand 11 alone) remains cumulative best** with non-overlapping
CI vs every other cell.

## The mid-tournament-vs-head-to-head reversal

The standing rule from 2026-05-26
(`feedback_use_headtohead_not_midtournament.md`) said mid-tournament
**understates** strong cells (Cand 11: 10.83% mid → 16.83% head-to-head).
Cell 6 shows mid-tournament can also **overstate**:

| Cell | mid-tournament ep10 | head-to-head (1200g) | Δ |
|---|---:|---:|---:|
| Cell 5 v2 (Cand 11 alone) | 10.83% | **16.83%** | +6.0pp (understated) |
| Cell 6 (stack) | 15.0% | **8.92%** | **−6.1pp (overstated)** |

**Both directions are real.** Mid-tournament is not a reliable proxy for
relative strength; only head-to-head among GNN cells is.

## Why the stack underperformed in head-to-head

Hypothesis (not yet confirmed by behavioral analysis): Cand 8+10's
dev-card-spam mechanism (documented in earlier Cell 1 journal: BuyDevCard
16.37/100 turns, ~3x Cand 11's rate) compounds with Cand 11's
expansion-heavy strategy in a way that's invisible against weak
opponents (Random + GnnMcts in mid-tournament) but exposed against real
GNN competition (Cell 5 v2 + Cell 1 in head-to-head).

When Cell 6 plays against Random + GnnMcts (mid-tournament), Random and
GnnMcts collectively take ~3% of games. Cell 6 doesn't have to be
maximally efficient — it just has to outplay them. Lookahead takes ~82-87%
of the remaining games, leaving ~10-15% for Cell 6 — that's the
mid-tournament number.

When Cell 6 plays against Cell 5 v2 + Cell 1 (head-to-head), those two
cells take ~24% of games (vs Random + GnnMcts's ~3% before). Cell 6's
share drops from ~15% to ~9% — Cell 5 v2 eats the difference. The stack's
policy is consistent (per-rotation 27–30, tight) but not strong enough
to win against Cell 5 v2's Cand-11-pure expansion strategy.

## Decision matrix update

| Cell | Mid-tournament best | Head-to-head best (n=1200) | Status |
|---|---:|---:|---|
| Cell 0 (vanilla) | 12.5% ep5 | 7.92% (vs LookV3 + Cell 1 + Cell 5 v2) | superseded |
| Cell 1 (Cand 8+10) | 10.83% ep5 | 8.75% / 8.83% (two runs) | superseded |
| Cell 2 (Cand 7+8+10) | killed | — | rejected |
| **Cell 5 v2 (Cand 11 alone)** | 10.83% ep10 | **14.67% / 16.83%** (two runs) | **cumulative best, reconfirmed** |
| **Cell 6 (Cand 11 + Cand 8+10 stack)** | **15.8% ep15** | **8.92%** | **superseded — stack adds no value over Cand 8+10 alone** |

## What this means for the loss-aug roadmap

1. **Stacking different action-class priors does NOT compound additively.**
   We assumed: Cand 11 fixes positioning (roads) + Cand 8+10 fixes
   closeout (cities) = both work together. Evidence says no — the
   combined policy is no better than Cand 8+10 alone, and meaningfully
   worse than Cand 11 alone in head-to-head.

2. **The "closeout bottleneck" framing from the diagnostic was correct
   but the fix wasn't.** Cand 11's 0.271 cities/turn-with-resources IS
   the bottleneck vs LookV3's 0.510, but Cand 8+10 doesn't actually
   address it for the Cand 11 policy — it just imports Cell 1's
   dev-card-spam habit.

3. **Cell 5 v2 reconfirmed at 14.67% in a second head-to-head** (vs
   16.83% in the first). The two runs differ by 2pp at n=1200, within
   the ±2pp 95% CI. Cand 11 alone's strength is statistically robust.

## Open question — what's the right next intervention?

The closeout gap (Cand 11's 27.1% vs LookV3's 51.0% city build-rate when
resources allow) remains. We thought Cand 8+10 would close it. It didn't.
Three options not yet tested:

1. **Closeout-only prior** — boost BuildCity specifically when (a)
   resources ≥ 3 ore + 2 wheat, (b) the player owns a settlement at a
   high-pip vertex. No general VP-class prior; just the literal "build
   the city you can build right now" signal. Different from Cand 8 which
   blanket-boosts all VP-yielding actions.

2. **Different λ_vp** — Cand 8 at λ_vp=0.10 inherits Cell 1's
   dev-card-spam. Try λ_vp=0.05 (half) or λ_vp=0.02 (very gentle) on
   top of Cand 11 — see if a weaker VP signal lifts city-rate without
   the BuyDevCard side-effect.

3. **Modified Cand 8** — patch `CLASS_VP_VALUE[BuyDevCard]` from 0.20 to
   0.05 in `action_classes.py`. Removes the dev-card-spam degenerate
   equilibrium I diagnosed earlier (see `feedback_*` memory). Combine
   with Cand 11. This was the most concrete fix idea from earlier
   sessions.

## Cited artefacts

- Cell output: `runs/v3/training/loss_aug/06_cand11_cand8_cand10_h128_l4/`
- Launch log: `cell6_launch.log`
- All 15 checkpoints: `training_h128_l4/checkpoint_epoch{01..15}.pt`
- Mid-tournament parquets: `training_h128_l4/mid_tournaments/2026-05-26T*/`
- 1200-game head-to-head: `runs/v3/tournaments/e10c_cell6_1200_2026_05_27/2026-05-27T16-33-e10c_triple_gnn/`
- Cell 5 v2 head-to-head (cumulative best, 16.83%): `runs/v3/tournaments/e10c_4way_1200_2026_05_26/2026-05-26T15-59-e10c_triple_gnn/`
- Smoke test: `mcts_study/tests/test_cell6_smoke.py` (commit `f905930`)
- Pre-launch GPU timing: `mcts_study/scratch_cell6_timing.py` (gitignored;
  showed 1.06× vanilla)

## Memory items to update

- `project_cand11_cumulative_best_2026_05_26.md` is **still current** — no
  change; Cell 5 v2 reconfirmed at 14.67% in this run. Add a note that
  Cell 6 was tested and did not displace it.
- `feedback_use_headtohead_not_midtournament.md` should be **strengthened**:
  mid-tournament can both understate AND overstate by ~6pp. Always use
  head-to-head.

## Conclusion

**Cell 6 = stacked Cand 11 + Cand 8 + Cand 10 does NOT beat Cell 5 v2
(Cand 11 alone)** in head-to-head. 8.92% vs 14.67% at n=1200 with
non-overlapping CIs; the gap is decisive.

The stack hypothesis ("compound positioning + closeout") failed
empirically. The mid-tournament metric strongly suggested otherwise
(15.8% at ep15 was the highest in the roadmap), so this is also a
proof point that mid-tournament misleads in either direction —
underrating Cand 11 alone, overrating Cand 11 + Cand 8+10.

**Cumulative best remains Cell 5 v2 ep10 (Cand 11 alone).** Future
candidates aimed at the city-closeout gap should NOT inherit Cand 8's
generic VP-class prior (which over-pulls BuyDevCard). Either patch
that prior or design a closeout-specific signal.
