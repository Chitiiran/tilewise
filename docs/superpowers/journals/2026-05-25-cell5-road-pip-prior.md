# Cell 5: Cand 11 (road-pip prior) — standalone experiment

**Date:** 2026-05-25
**Plan:** `docs/superpowers/plans/2026-05-25-road-pip-prior.md`
**Spec sections:** Mathematical Specification (locked); Layer-1 KL, Gate A, λ_road=0.05.
**Status:** v1 KILLED (perf regression); v2 RUNNING — launched 2026-05-25 15:56 UTC with vectorized impl + per-batch observability, PID 584
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

Ran 2026-05-25, ~40 min cache load + ~5 min walk on 1000 random samples
from `~/catan_cache/cache_100k.pt` (3,219,479 positions, 3 chunks, 29 GB).

```
=== Cand 11 calibration on 1000 random cache samples ===
Samples with NO legal settlement (Gate A part 1): 893 (89.3%)
  Of those, with at least one nonzero road score (Gate A fully fires): 195 (21.8%)
  All-zero road scores (gate part 3 blocks): 698 (78.2%)
  OVERALL gate-firing rate: 195/1000 = 19.5%

|L_R| histogram on Gate-A-part-1 samples:
  |L_R| =  0:  618 samples
  |L_R| =  2:   16 samples
  |L_R| =  3:   52 samples
  |L_R| =  4:    4 samples
  |L_R| =  5:   15 samples
  |L_R| =  6:   27 samples
  |L_R| =  7:   49 samples
  |L_R| =  8:   63 samples
  |L_R| =  9:   34 samples
  |L_R| = 10:   12 samples
  |L_R| = 11:    3 samples

Mean prior entropy (firing samples): 0.747
Mean MCTS-visits entropy over legal roads (firing samples): 1.929
Ratio prior/visits: 0.387
  (1.0 = comparable sharpness; <0.5 = prior much sharper, consider lower lambda_road)
```

### Key numbers

| Metric | Value | Plan band (per plan §9.2) | Status |
|---|---:|---:|---|
| Overall gate-firing rate | **19.5%** | 5-60% | ✅ in-band |
| Prior/visits entropy ratio | **0.387** | 0.3-2.0 | ⚠ in-band but at low end (< 0.5 = prior much sharper than visits, per script) |
| \|L_R\|=0 when no settlement | 618 / 893 (69%) | — | ℹ unexpected — most no-settlement states have no roads either (post-roll, post-robber, EndTurn-only) |

### Interpretation

1. **Gate A fires on 19.5% of samples** — comfortably in the design band. About 1 in 5 training samples gets a Cand 11 gradient signal. Enough to produce a learnable effect over 15 epochs × ~10k batches/epoch.

2. **The prior is sharper than MCTS visits** (entropy 0.747 vs 1.929; ratio 0.387). With λ_road=0.05 this means the KL pulls the policy toward a near-one-hot target (highest-pip road) while MCTS visits spread mass across multiple roads. At the planned λ_road=0.05 this is a fairly aggressive pull.

3. **78% of no-settlement samples have no useful road choice either** — they're forced-action states (post-roll, post-robber, EndTurn-only). Gate A's `has_legal_road + has_score` checks correctly skip them.

### Decision: λ_road = 0.05 (chosen 2026-05-25)

User chose 0.05 after reviewing the trade-off table. Rationale per chat:
the script's <0.5 entropy ratio threshold is a heuristic; the plan's
acceptance band (0.3-2.0) covers 0.387; and Cand 11's effective gradient
share (λ × gate-firing rate) at 0.05 is **0.05 × 0.195 ≈ 0.010**, already
7× weaker than Cand 8's effective share in Cell 1 (0.10 × 0.70 ≈ 0.070).
Going lower risked a null Cell 5 (uninformative, costs 42h to resolve).

## Launch (2026-05-25 10:06 UTC)

PID 569, detached. Launch command:

```bash
nohup python scripts/train_grid_inproc.py \
  --cache-path ~/catan_cache/cache_100k.pt \
  --out-root runs/v3/loss_aug/05_cand_road_pip_h128_l4 \
  --status-file runs/v3/dashboard/cell5.json \
  --epochs 15 --batch-size 256 --device auto \
  --rotate --rotate-mode random --cells h128_l4 --seed 0 \
  --mid-tournament-every 5 \
  --lambda-road 0.05 \
  > runs/v3/loss_aug/05_cand_road_pip_h128_l4/cell5_launch.log 2>&1 &
```

Expected timeline (per Cell 1 wall-clock 19.3h):
- Cache load: ~40 min
- ep1-5: ~5h
- ep5 mid-tournament: ~45 min → **first decision point**
- ep5-10: ~5h
- ep10 mid-tournament: ~45 min
- ep10-15: ~5h
- ep15 mid-tournament: ~45 min
- Total: ~17-18h to completion

**Early-kill rule (per plan §10.3):** if ep5 PureGnn ≤ 12 wins out of 120
(≤10.0%, which is ≥1.5pp below Cell 0's 12.50% / 15 wins), kill the run
and journal the result.

## v1 → v2 — fix landed (2026-05-25 same day)

v1 PID 569 was killed at ~5h elapsed without completing epoch 1, due to
a per-sample Python loop in `road_pip_prior` that caused ~40× slowdown
on GPU via CUDA pipeline stalls. See
`docs/superpowers/journals/2026-05-25-cand11-perf-rca.md` for the
evidence-based RCA and verified fix.

Fix commits (all on v3):
  - `5e311eb` feat(observability): per-batch progress + mid-epoch dashboard writes
  - `fe72f4e` perf(cand11): batched road_pip_prior — eliminate per-sample Python loops
  - `699aa8a` test(cand11): equivalence — 100 random samples + B=8 grad within 1e-6
  - `540d6c4` docs(rca): post-fix verification

Post-fix measured GPU per-batch (e1 fixture, B=256, real h128_l4 GNN):
  - vanilla: 70.5 ms/batch
  - Cand 11: 75.3 ms/batch (+7% overhead)
  - Projected: 15.8 min/epoch, ~4h for full 15-epoch run.

## v2 launch (2026-05-25 15:56 UTC)

PID 584, detached. Output dir `runs/v3/loss_aug/05_cand_road_pip_h128_l4_v2/`
(v1 dir preserved as evidence). Same launch command as v1 except output
paths suffixed `_v2`.

Expected timeline:
  - Cache load: ~40 min
  - ep1 boundary: ~56 min post-launch (16 min training)
  - ep5 mid-tournament: ~2.4h post-launch
  - Full run: ~4h post-cache-load = ~4.7h total

With per-batch observability, the launch log now emits a line every ~20s
showing batch index, loss, ms/batch, and ETA. Any pathological slowness
or wedge will be visible within ~5 min of training start.

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
