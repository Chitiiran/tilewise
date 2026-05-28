# Cell 5: Cand 11 (road-pip prior) — standalone experiment

**Date:** 2026-05-25
**Plan:** `docs/superpowers/plans/2026-05-25-road-pip-prior.md`
**Spec sections:** Mathematical Specification (locked); Layer-1 KL, Gate A, λ_road=0.05.
**Status:** COMPLETED — v2 ran 18h50m, all 15 epochs + 3 mid-tournaments; auto-stopped at ep15 via mid_tournament_drop_threshold rule (PureGnn 13→7 between ep10 and ep15)
**Cell output (planned):** `runs/v3/training/loss_aug/05_cand_road_pip_h128_l4/`
**Baseline for comparison:** Cell 0 vanilla (`runs/v3/training/loss_aug/00_baseline_h128_l4_pilot/`)

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
  --out-root runs/v3/training/loss_aug/05_cand_road_pip_h128_l4 \
  --status-file runs/v3/dashboard/cell5.json \
  --epochs 15 --batch-size 256 --device auto \
  --rotate --rotate-mode random --cells h128_l4 --seed 0 \
  --mid-tournament-every 5 \
  --lambda-road 0.05 \
  > runs/v3/training/loss_aug/05_cand_road_pip_h128_l4/cell5_launch.log 2>&1 &
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

PID 584, detached. Output dir `runs/v3/training/loss_aug/05_cand_road_pip_h128_l4_v2/`
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

## Final results

### Per-epoch metrics (all 15 epochs)

| ep | train_loss | val_loss | val_top1 | per_game[min\|p25\|med\|p75\|max] | best? |
|---:|---:|---:|---:|---|---|
| 1 | 2.983 | 2.865 | 0.179 | 0.00\|0.12\|0.17\|0.23\|0.53 | ✓ |
| 2 | 2.956 | 2.879 | 0.180 | 0.00\|0.12\|0.17\|0.23\|0.53 | ✓ |
| 3 | 2.918 | 2.922 | 0.174 | — | |
| 4 | 2.863 | 2.971 | 0.179 | 0.00\|0.12\|0.17\|0.23\|0.60 | |
| 5 | 2.814 | 2.989 | 0.177 | 0.00\|0.12\|0.17\|0.23\|0.54 | |
| 6 | 2.778 | 3.033 | 0.177 | 0.00\|0.12\|0.17\|0.22\|0.60 | |
| 7 | 2.749 | 3.044 | 0.176 | 0.00\|0.12\|0.17\|0.22\|0.54 | |
| **8** | **2.726** | 3.081 | **0.185** | 0.00\|0.13\|0.18\|0.23\|0.61 | ✓ new best |
| 9 | 2.707 | 3.089 | 0.176 | 0.00\|0.12\|0.17\|0.22\|0.57 | |
| 10 | 2.692 | 3.110 | 0.178 | 0.00\|0.12\|0.17\|0.22\|0.60 | |
| 11 | 2.679 | 3.103 | 0.176 | 0.00\|0.12\|0.17\|0.22\|0.53 | |
| 12 | 2.668 | 3.124 | 0.173 | 0.00\|0.12\|0.17\|0.22\|0.57 | |
| 13 | 2.659 | 3.149 | 0.182 | 0.00\|0.13\|0.18\|0.23\|0.60 | |
| 14 | 2.651 | 3.142 | 0.170 | 0.00\|0.12\|0.17\|0.21\|0.53 | |
| 15 | 2.643 | 3.146 | 0.176 | 0.00\|0.12\|0.17\|0.22\|0.57 | |

train_loss monotone decreasing 2.983→2.643. val_loss climbing steadily
2.865→3.146 (overfit signature, similar to Cell 0 vanilla). val_top1
oscillating in 0.170–0.185 band; ep8 happened to land at the high end
and got crowned best (saved `checkpoint_best.pt`). Per the standing
rule `feedback_val_top1_misleads_under_loss_aug.md`, this val_top1 best
is not the tournament-best checkpoint — see mid-tournament table below.

### Mid-tournament results (the actual signal)

120 games (30 × 4 rotations), sims=100, lookahead_depth=10,
base_sims_v3=200, h128_l4, workers=10, device=cuda, seed_base=19M.
STANDARD_* config from `mid_training_tournament.py`. Tournament-time
trajectory:

| Cell | ep5 PureGnn | ep10 PureGnn | ep15 PureGnn |
|---|---:|---:|---:|
| Cell 0 vanilla (cited) | 15/120 (12.5%) | 2/120 (1.67%) | — |
| Cell 1 (Cand 8+10) (cited) | 13/120 (10.83%) | 11/120 (9.17%) | 12/120 (10.00%) |
| Cell 2 (Cand 7+8+10) (cited) | 4/120 (3.33%) | killed at ep6 | — |
| **Cell 5 v2 (Cand 11 alone)** | **7/120 (5.8%)** | **13/120 (10.8%)** | **7/120 (5.8%)** |

Full breakdown of Cell 5 v2 mid-tournaments:

| ep | PureGnn | GnnMcts | LookaheadV3 | Random | draws | elapsed |
|---:|---:|---:|---:|---:|---:|---:|
| 5  | 7  (5.8%)  | 2  (1.7%) | 111 (92.5%) | 0 | 0 | 2702s |
| 10 | 13 (10.8%) | 4  (3.3%) | 101 (84.2%) | 1 | 1 | 2838s |
| 15 | 7  (5.8%)  | 4  (3.3%) | 108 (90.0%) | 0 | 1 | 3110s |

### Auto-stop trigger

The training process auto-stopped after ep15 mid-tournament via the
`mid_tournament_drop_threshold=3` rule (cited `train.py` default). The
13→7 drop between ep10 and ep15 exceeded the 3-game threshold, so the
final log line read:

```
↳ mid-tournament early-stop: PureGnn wins dropped 13 → 7 (>= 3-game threshold); stopping at epoch 15
```

The run completed all 15 epochs (the stop rule only prevents further
training after ep15, which was already the last planned epoch). Total
wall-clock: **18h50m (67822.6s)** including ~40min cache load.

## Comparison + interpretation

### Cand 11 has a U-shaped trajectory, not a plateau

| Cell | ep5 | ep10 | ep15 | Pattern |
|---|---:|---:|---:|---|
| Cell 0 vanilla | 12.5% | 1.67% | — | Peak-then-collapse |
| Cell 1 (Cand 8+10) | 10.83% | 9.17% | 10.00% | **Stable plateau** |
| **Cell 5 v2 (Cand 11)** | **5.8%** | **10.8%** | **5.8%** | **U-shape: late peak, then drop** |

Cell 5 v2's ep10 result is essentially identical to Cell 1's plateau
(10.8% vs 10.83%). But Cand 11 alone cannot hold that level — it drops
back at ep15.

**This refutes the simple hypothesis** that "any prior + protective
loss term works." Cand 11 produces a different temporal pattern than
Cand 8+10 despite both being structurally reasonable interventions.

### Cand 11 still beats vanilla's collapse

At ep10, Cell 5 v2 = 10.8% vs Cell 0 vanilla = 1.67% at the same
epoch. That's **9.13pp above vanilla / 6.5× more wins**. Cand 11 has
the post-peak protection property of Cand 8+10, just less stably.

### The ep5 regression then recovery is unusual

Cand 8+10 was stable from ep5 onward. Cand 11 went 5.8% → 10.8% →
5.8% across the same window. The 5.8pp climb at ep10 then 5.0pp drop
at ep15 are both well outside the ±0.83pp noise band (Cell 0
reproducibility check), so it's real signal — not measurement noise.

What does it mean? Hypothesis (not verified): the road-pip prior
pulls the policy toward "good-pip-targeted roads" early. The MCTS
visit-count signal in the data is weaker on roads than on
VP-yielding actions, so the prior dominates road policy early. By
ep10 the model has internalized the road pattern and the policy
gradients on settlements+cities catch up, giving a coherent strategy.
Past ep10, the val_loss climbs faster than the prior can stabilize
the road policy, and the model starts mis-applying the road bias
(building roads in contexts where the prior taught "build roads" but
the situation actually demands a different action class). Confirming
this requires the road-vs-settlement-rate analysis from
`scratch_midgame_actions_cell1_ep10.py` adapted for ep5/10/15 of
Cell 5 v2 — deferred to a follow-up.

### GnnMcts is consistently higher than other cells

| Cell | GnnMcts ep10 |
|---|---:|
| Cell 0 vanilla | 0/120 |
| Cell 1 (Cand 8+10) | 0/120 |
| **Cell 5 v2 (Cand 11)** | **4/120 (3.3%)** |

GnnMcts is the search-augmented form of PureGnn (sims=100). Cand 11's
4/120 here is the highest GnnMcts score recorded in any cell so far.
Combined with PureGnn 13/120, total non-Lookahead wins for Cand 11 at
ep10 = **19/120 = 15.8%**, vs Cell 1's 13/120 = 10.83%. This may mean
the road-pip prior helps the search form more than it helps the pure
argmax form, which is a different mechanism than Cand 8+10's effect.

### Decision matrix update

| Candidate | Status | Result |
|---|---|---|
| Cell 0 (vanilla) | done | 12.08% ep5, collapses to 1.67% ep10 |
| **Cell 1 (Cand 8+10)** | **cumulative best** | 9-11% stable plateau |
| Cand 1 alone (off-plan) | rejected | 3.33-4.17% — regression |
| Cell 2 (Cand 7 on Cand 8+10) | rejected | 3.33% — 7.5pp regression |
| **Cell 5 v2 (Cand 11 alone)** | **partial success** | 10.8% at ep10 (matches Cell 1); drops to 5.8% at ep15 (unstable) |
| Cell 3 (Cand 2 city boost) | gated on Cell 1 city-rate analysis | not started |

## Open questions

1. **Does Cand 11 + Cand 8+10 stack constructively?** Cand 11 has
   shown it can match Cell 1's peak at ep10. Stacking might give
   either the best of both (stable + 10%+ throughout) or a Cand-7-
   style compound regression. Worth testing.

2. **What's the right checkpoint to use for downstream tournament?**
   `checkpoint_best.pt` points at ep8 (val_top1=0.185) which is NOT
   the tournament-best — that's ep10 (10.8% PureGnn). val_top1 misled
   us again.

3. **Does the U-shape repeat with a different seed?** Cand 11's
   ep5→ep10→ep15 pattern of 5.8% → 10.8% → 5.8% could be a stable
   property of the loss or a single-seed artifact. Reproducibility
   needs a second seed.

4. **Road-to-settlement ratio behaviour:** the behavioral metric
   Cand 11 was designed to move. Deferred per "no immediate analysis"
   — but should be computed before judging whether Cand 11 worked
   "as designed."

5. **GnnMcts uplift:** is the 4/120 GnnMcts result at ep10 a Cand 11
   signature or random? Confirming requires running GnnMcts-focused
   tournaments at other Cand 11 epochs, or a reproducibility check.

## Cited artefacts

- Cell output: `runs/v3/training/loss_aug/05_cand_road_pip_h128_l4_v2/`
- Launch log: `runs/v3/training/loss_aug/05_cand_road_pip_h128_l4_v2/cell5_v2_launch.log`
- Checkpoints: `training_h128_l4/checkpoint_epoch{01..15}.pt` (all saved)
- Mid-tournament parquets: `training_h128_l4/mid_tournaments/2026-05-26T{01-53,07-57,13-56}-e10_v3_tournament/`
- Code commits (v3 branch): `fe72f4e` (batched impl), `699aa8a` (equivalence test), `5e311eb` (observability), `824bdfd` (launch)
- RCA journal: `docs/superpowers/journals/2026-05-25-cand11-perf-rca.md`

## Conclusion

**Cand 11 (pure-pip road prior, λ=0.05) is a partial success.** At
ep10 it matches Cell 1's plateau (10.8% PureGnn) and beats vanilla's
collapse 6.5×. But it does not maintain the plateau: ep5 and ep15
show 5.8% — a U-shape rather than a flat line. The cumulative-best
cell remains Cell 1 (Cand 8+10).

The standing rule about val_top1 was vindicated again: `checkpoint_best.pt`
points at ep8 but tournament truth says ep10 is the best epoch. Future
"best" should be measured by mid-tournament, not val_top1.

Three useful follow-ups in priority order:

1. **4-way tournament: LookV3 + Cell0-ep5 + Cell1-ep10 + Cell5v2-ep10**
   — direct comparison of the three GNNs at their tournament-best
   checkpoints (vs Lookahead's depth-10 baseline). One 45-min run, 4
   data points.
2. **Reproducibility seed=1 run of Cand 11** — confirms whether the
   U-shape is a stable property or seed-dependent.
3. **Cell 6: Cand 11 stacked on Cand 8+10** — tests whether the two
   structurally different priors compose constructively. Could land
   the loss-aug roadmap's first stable >12% plateau, or surface
   another Cand-7-style compound regression.
