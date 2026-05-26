# Cell 6: Cand 11 + Cand 8 + Cand 10 stacked

**Date:** 2026-05-26
**Status:** RUNNING — launched 2026-05-26 19:37 UTC, PID 4702
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
  --out-root runs/v3/loss_aug/06_cand11_cand8_cand10_h128_l4 \
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

(populate as training progresses)

| ep | train_loss | val_loss | val_top1 | vp_swap | best? |
|---:|---:|---:|---:|---:|---|

## Mid-tournament results

(populate after each mid-tournament)

| ep | PureGnn | GnnMcts | LookV3 | Random | draws |
|---:|---:|---:|---:|---:|---:|

## Behavioral analysis at ep5

(populate after ep5 mid-tournament — same format as Cell 5 v2 analysis)

| Role | roads/100 | settle/100 | cities/100 | dev_card/100 | trade_propose% |
|---|---:|---:|---:|---:|---:|
| LookV3 reference | 17.48 | 3.56 | 7.30 | 10.85 | 32% |
| Cell 5 v2 (Cand 11) ref | 22.06 | 5.74 | 4.86 | 5.77 | 34% |
| Cell 1 (Cand 8+10) ref | 16.64 | 2.90 | 1.09 | 16.37 | 17% |
| **Cell 6 (stack)** | | | | | |

## Final head-to-head (post-ep15)

(populate after 1200-game e10c run vs Cell 5 v2 ep10 + Cell 1 ep10 + LookV3)

| Player | Wins / 1200 | % | vs Cell 5 v2 (16.83%) |
|---|---:|---:|---:|

## Conclusion

(fill in)
