# Cell 5: Cand 11 (road-pip prior) — standalone experiment

**Date:** 2026-05-25
**Plan:** `docs/superpowers/plans/2026-05-25-road-pip-prior.md`
**Spec sections:** Mathematical Specification (locked); Layer-1 KL, Gate A, λ_road=0.05.
**Status:** STUB — code complete, awaiting calibration + launch decision
**Cell output (planned):** `runs/v3/loss_aug/05_cand_road_pip_h128_l4/`
**Baseline for comparison:** Cell 0 vanilla (`runs/v3/loss_aug/00_baseline_h128_l4_pilot/`)

## Setup

| Setting | Cell 0 (baseline) | Cell 5 (this run) |
|---|---|---|
| Architecture | h128_l4 (632k params) | h128_l4 (same) |
| Cache | cache_100k.pt | same |
| Batch size | 256 | 256 |
| LR | 1e-3 (Adam) | same |
| Augmentation | random hex rotation | same |
| Seed | 0 | 0 |
| `lambda_vp` (Cand 8) | 0.0 | 0.0 |
| `vp_compare_rule` (Cand 10) | False | False |
| **`lambda_road` (Cand 11)** | **0.0** | **0.05** |

## Implementation commits (v3 branch)

- `3b425de` feat(cand11): road-pip prior topology helpers
- `298262e` feat(cand11): far_endpoint + settlement_legal_mask
- `60bca22` feat(cand11): per-sample road score + linear prior target
- `2b3a0f0` feat(cand11): batched road_pip_prior_loss with Gate A + Layer-1 KL
- `d8b0d70` feat(cand11): plumb lambda_road through train_main
- `fb7c029` feat(cand11): add --lambda-road CLI flag to train_grid_inproc.py
- `10e916c` test(cand11): Cell 5 smoke test

12 unit tests + 2 smoke tests, all green.

## Calibration result (Task 7, pre-launch)

(paste output of scratch_road_pip_calibration.py here once it completes)

```
TODO: fill in once calibration finishes
```

Key numbers:
- Gate-firing rate: __TODO__%
- Prior/visits entropy ratio: __TODO__

**Acceptance band (per plan):** gate-firing 5-60%, entropy ratio 0.3-2.0.
**Decision:** TODO (launch / revise λ_road / abort).

## Per-epoch metrics

(fill in as training progresses; format mirrors Cell 1 journal)

| ep | train_loss | val_loss | val_top1 | best |
|---:|---:|---:|---:|---|

## ep5 mid-tournament — the decision point

(fill in after ep5 mid-tournament completes — typically ~5h after launch)

| Player | Cell 5 | Cell 0 baseline | Δ |
|---|---:|---:|---:|
| PureGnn | / 120 ( %) | 15 / 120 (12.50%) | |
| GnnMcts | / 120 ( %) | 2 / 120 (1.67%) | |
| LookaheadMctsV3 | / 120 ( %) | 102 / 120 (85.00%) | |
| Random | / 120 ( %) | 1 / 120 (0.83%) | |

**Decision rule (per plan):** if PureGnn ≥ 1.5pp below Cell 0's ep5 (i.e. ≤10.58%
= ≤12 wins/120), kill the run, journal the result, do not continue. Otherwise,
let it run to ep15.

## Behavioral metric — road-to-settlement ratio

(populate after running scratch_midgame_actions_cell1_ep10.py adapted for Cell 5)

| Role | roads/100 turns | settlements/100 turns | roads ÷ settlements |
|---|---:|---:|---:|
| Cell 1 PureGnn ep10 (cited prior journal) | 17.49 | 2.43 | 7.2 |
| Cell 5 PureGnn ep5 (this) | | | |
| Lookahead in-tournament ep10 (cited) | 18.10 | 4.53 | 4.0 |

The metric Cand 11 is designed to move: if PureGnn's roads-per-settlement drops
from 7.2 toward Lookahead's 4.0, Cand 11 is working as designed even if the
overall winrate doesn't move yet.

## Conclusion

(fill in)
