# Cell 2: Cand 7 stacked on Cand 8 + Cand 10 — REGRESSION

**Date:** 2026-05-25
**Plan:** `C:\Users\chiti\.claude\plans\let-s-read-for-context-humming-storm.md` (Cell 2)
**Spec:** `docs/superpowers/specs/2026-05-09-loss-augmentation-design.md` (Candidate 7)
**Cell 2 output:** `runs/v3/loss_aug/04_cand7_on_cand8_10/`
**Cell 1 baseline (cumulative best):** `runs/v3/loss_aug/01_cand8_cand10_h128_l4/`
**Commit:** `3b0eacc` (feat: Cand 7 — action-class-balanced policy CE)
**Status:** KILLED at epoch 6 after ep5 mid-tournament showed strong regression.

## Headline

**Cand 7 (action-class-balanced policy CE) stacked on Cand 8 + Cand 10 produces a 7.5pp PureGnn winrate REGRESSION at epoch 5 vs Cell 1 baseline.**

| Cell | Loss config | ep5 PureGnn winrate | Δ vs Cell 1 |
|---|---|---:|---:|
| Cell 0 (vanilla) | none | 12.08–12.50% (n=2 avg) | +1.25pp |
| **Cell 1 (cumulative best)** | Cand 8 + Cand 10 | **10.83%** | (baseline) |
| **Cell 2 (this run)** | Cand 7 + Cand 8 + Cand 10 | **3.33%** | **−7.50pp** |

The result is **far outside the ±1-game (±0.83pp) FP-non-determinism noise band** verified for Cell 0. Cand 7 stacked on top of the cumulative best is a **net negative** intervention; rolled back, not adopted.

## Setup

| Setting | Cell 1 (baseline) | Cell 2 |
|---|---|---|
| Architecture | h128_l4 (632k params) | h128_l4 (same) |
| Cache | `~/catan_cache/cache_100k.pt` (3.22M positions) | same |
| Batch size | 256 | 256 |
| LR | 1e-3 (Adam) | same |
| Augmentation | random hex rotation | same |
| Seed | 0 | 0 |
| `lambda_vp` (Cand 8) | 0.10 | 0.10 |
| `vp_compare_rule` (Cand 10) | True | True |
| **`class_balanced_policy` (Cand 7)** | **False** | **True** |
| Mid-tournaments | every 5 epochs, 120 games vs LookaheadV3, seed_base=19M, workers=10, cuda | same |

Cand 7 = per-sample action-class-balanced policy target. For each legal action `a` in a sample, divide its teacher-target weight by the number of legal action_ids in the same action_class, then renormalize so the per-sample target sums to 1. Applied **after** Cand 10's vp_compare swap. Val loss uses the rebalanced target; `val_top1` uses the original teacher target so cross-cell numbers remain comparable.

## Why this was the planned next step

Per the loss-augmentation spec (lines 391–415) and plan ordering doc (lines 137, 234–254):

- Cited training-data → tournament behavior gap: training data has ~16% road action rows, ~7.6% settle rows; PureGnn at tournament only produces ~4.4% road and ~1.4% settle actions (cited `scratch_check_setup_samples.py` + `scratch_midgame_actions.py`).
- Hypothesis: ProposeTrade occupies 20 action_ids and BuildRoad 72, so cross-entropy across the 280-dim policy gives high-id-count classes more total gradient share regardless of MCTS preference. Class-balancing was supposed to equalize per-class gradient share.

## Per-epoch metrics — Cell 2 vs Cell 1

| ep | Cell 2 train_loss | Cell 2 val_loss | Cell 2 val_top1 | Cell 1 train_loss | Cell 1 val_top1 |
|---:|---:|---:|---:|---:|---:|
| 1 | **2.520** | 2.869 | **0.188** | 2.739 | 0.173 |
| 2 | **2.481** | 2.932 | **0.189** | 2.703 | 0.166 |
| 3 | **2.422** | 2.936 | 0.188 | 2.645 | 0.181 |
| 4 | **2.354** | 3.025 | 0.189 | 2.574 | 0.168 |
| 5 | **2.303** | 3.042 | 0.187 | 2.522 | 0.170 |

**Pre-tournament reading (misleading):** Cell 2's train_loss was ~0.22 lower at every epoch; val_top1 held at 0.188–0.189 vs Cell 1's wobbly 0.166–0.181. This looked encouraging.

**The trap:** the journal for Cell 1 already documented that *"val_top1 is uninformative in this regime — the policy is being pulled toward VP-economy targets and away from raw MCTS visit counts."* The reading I gave during the run treated higher val_top1 as a positive signal, when the spec and journal had explicitly warned it was not a reliable proxy for downstream tournament winrate. **This is the same trap Cell 1 had already named, and we walked into it again.**

`vp_swap` rate stayed flat at 18.67–18.69% across all 5 epochs — Cand 7 did not interfere with Cand 10's swap mechanics, as expected. So Cand 10 is structurally unaffected by the rebalance; the damage is in how Cand 7 distorts what the supervised CE is reinforcing.

## ep5 mid-tournament — the actual signal

120 games, 30 × 4 rotations, sims=100, lookahead_depth=10, base_sims_v3=200, h128_l4, workers=10, device=cuda, seed_base=19M. Identical config to Cell 0/Cell 1 mid-tournaments.

| Player | Cell 2 (this) | Cell 1 baseline | Cell 0 vanilla |
|---|---:|---:|---:|
| **PureGnn** | **4 / 120 (3.33 %)** | 13 / 120 (10.83%) | 15 / 120 (12.50%) |
| GnnMcts | 1 / 120 (0.83%) | 2 / 120 (1.67%) | 2 / 120 (1.67%) |
| LookaheadMctsV3 | 113 / 120 (94.17%) | 105 / 120 (87.50%) | 102 / 120 (85.00%) |
| Random | 2 / 120 (1.67%) | 0 / 120 (0.00%) | 1 / 120 (0.83%) |
| draws (no winner) | 0 / 120 | 0 / 120 | 0 / 120 |

Tournament elapsed: 2855 s (47.6 min) — within normal range.

## Why Cand 7 likely backfired

The design intent — "equalize per-class gradient share" — is structurally at odds with how the MCTS visit-count target works:

1. **MCTS visits already encode correctness**, not class density. When the teacher concentrates visits on a single road, the supervised CE correctly reinforces that road. Cand 7 then divides that target by ~5–10 (number of legal roads), attenuating the road signal that the teacher had identified as correct.
2. **Conversely**, when only 1 BuildCity action is legal, Cand 7's denominator is 1 → the city target is preserved. Combined with renormalization, sparse-class actions get **amplified** relative to dense-class actions, even when the teacher didn't visit them heavily.
3. **Compound with Cand 8** (action-class VP prior, which already pulls policy toward VP-yielding actions): Cand 7 amplifies whatever class has 1 legal action — typically BuildCity / BuildSettlement / PlayVpCard — at the cost of the road class that the policy needed to learn properly. The model now over-prefers VP-grant moments and under-trains the road decisions that *enable* those moments.

This is consistent with the headline numbers: PureGnn finishes games less often (3.3% vs 10.8%) because the policy lost competence on road decisions, which is the resource-positioning that has to precede the VP-grant moves the model is now over-attracted to.

## Statistical confidence

Cell 0's reproducibility check (verified 2026-05-11): byte-identical at ep10 across two independent runs. Noise band = ±1 game out of 120 (±0.83pp).

Cell 2's ep5 result (4/120) is **9 games below** Cell 1's ep5 (13/120) — **11× the noise band**. The regression is unambiguous, not a fluke.

## Conclusions

1. **Cand 7 stacked on Cand 8 + Cand 10 is a regression at ep5. Roll back; do not adopt.** The intervention runs counter to the supervised signal the teacher provides.

2. **Cumulative best remains Cell 1 (Cand 8 + Cand 10).** No update to the cumulative-best baseline.

3. **`val_top1` deceived us again.** Cell 2's training metrics looked strictly better than Cell 1's. Both the spec and the prior journal already documented this is not a reliable signal. **Future cells: do not call a result "encouraging" based on train metrics — wait for the tournament. The mid-tournament infrastructure is the only honest signal.**

4. **The spec's hypothesis that class-count density is the bottleneck appears falsified for h128_l4 at 100k cache.** The 280-dim policy CE on MCTS visit counts is already a calibrated supervised signal; rebalancing by class count damages that calibration. If there is a real issue with the training-data → tournament behavior gap on roads, it must be addressed in a way that does not distort the teacher target.

5. **Compute cost of this null result: ~7h13m wall-clock.** Killed at epoch 6/15 after ep5 mid-tournament. Would have cost ~12 more GPU-hours to finish; killed early per "salvage compute" rule (cited `feedback_salvage_compute_and_time.md`).

## Open questions

1. **Is the class-density hypothesis dead for all sizes, or only h128_l4?** A smaller architecture might be more sensitive to gradient noise from class imbalance. Not on the priority path — defer.

2. **Could Cand 7 work in isolation (no Cand 8, no Cand 10)?** Possibly — the compound effect with Cand 8 may be what tipped it negative (per the "compound with Cand 8" hypothesis above). Untested. The smoke test `test_cand7_isolated` passed and ran without NaN, but only 1 epoch on the toy fixture. Not on the priority path — defer.

3. **Was the issue specifically in the train-time rebalance, the val-time rebalance, or both?** Implementation rebalances both. Could test "train-only rebalance, no val rebalance" but unlikely to flip the sign of a 7.5pp gap.

4. **Cell 3 next: Cand 2 (city-upgrade target boost) — should it run?** Per plan: only if Cell 1's city-rate didn't move toward Lookahead's. Decision requires running midgame-actions analysis on Cell 1's ep10 tournament parquets first.

## Cited artefacts

- Code: `mcts_study/catan_gnn/train.py::class_balanced_target` (commit 3b0eacc, lines ~94–135)
- Action class table: `mcts_study/catan_gnn/action_classes.py::ACTION_CLASS_ID, NUM_ACTION_CLASSES` (commit 3b0eacc)
- CLI flag: `mcts_study/scripts/train_grid_inproc.py::--class-balanced-policy` (commit 3b0eacc)
- Unit tests: `mcts_study/tests/test_action_class_balanced.py` — 9 tests, all green
- Smoke tests: `mcts_study/tests/test_cand7_stack.py` — 2 tests, all green
- Cell 2 training output: `runs/v3/loss_aug/04_cand7_on_cand8_10/training_h128_l4/`
- Cell 2 launch log: `runs/v3/loss_aug/04_cand7_on_cand8_10/cell2_launch.log`
- Cell 2 ep5 tournament parquets: `runs/v3/loss_aug/04_cand7_on_cand8_10/training_h128_l4/mid_tournaments/2026-05-13T22-12-e10_v3_tournament/` (10 workers × 4 rotations)
- Prior journal for comparison: `docs/superpowers/journals/2026-05-12-cell0-cell1-baseline-vs-cand8_10.md`

## Reproduction

```bash
# Cell 2 launch (this regression):
python scripts/train_grid_inproc.py \
  --cache-path ~/catan_cache/cache_100k.pt \
  --out-root runs/v3/loss_aug/04_cand7_on_cand8_10 \
  --status-file runs/v3/dashboard/cell3.json \
  --epochs 15 --batch-size 256 --device auto \
  --rotate --rotate-mode random --cells h128_l4 --seed 0 \
  --mid-tournament-every 5 \
  --lambda-vp 0.10 --vp-compare-rule \
  --class-balanced-policy
```

Killed at 07:11 wall-clock with SIGTERM; PID 550. Cache load 2484 s. Per-epoch ~63 min train + ~6 min val.

## Side issue: Cand 1 alone (off-plan) also regressed

Earlier in this work block, a Cand 1 alone run (`runs/v3/loss_aug/03_cand1_only_h128_l4/`) was launched with `--lambda-settle 0.20` and no Cand 8/10 stack. Trained ep1 + did ep1 mid-tournament (Phase A), then a chain-script was set up to resume to ep10 with ep5/ep10 tournaments. The chain script never fired — WSL was restarted before it triggered. After the restart, Phase B was relaunched directly (without the chain). It got to ep5 + did ep5 mid-tournament before being killed for this Cell 2 work.

Cand 1 alone results (also a regression vs Cell 1 baseline):

| Run | ep | PureGnn winrate |
|---|---|---:|
| Cand 1 alone | 1 | 4.17% (5/120) |
| Cand 1 alone | 5 | 3.33% (4/120) |

Mean opening pip analysis showed Cand 1 *did* shape openings as designed (PureGnn pip 18.45–18.64, the highest of any model tested), but conversion stayed at 4–6% — the model picks the best opening hex spots and then fails to close. The PureGnn opening-pip lead in 64–65 of 120 games converted to only 4 wins (6%), vs LookV3 converting 29–35/35 highest-pip leads (97–100%).

This confirms the spec's own risk #3 for Cand 1 (line 79): *"The conditional win-rate data suggests opening placement isn't the bottleneck. This loss may not help."* It didn't. Cand 1 alone is also rolled back; not adopted.

## Decision matrix update

| Candidate | Status | Result |
|---|---|---|
| Cell 0 (vanilla) | done | 12.08% ep5, collapses to 1.67% ep10 |
| **Cell 1 (Cand 8 + Cand 10)** | **adopted as cumulative best** | 9.17–10.83% plateau ep5/10/15 |
| Cand 1 alone (off-plan) | rejected | 3.33–4.17% — regression |
| **Cell 2 (Cand 7 on Cand 8+10)** | **rejected** | 3.33% — 7.5pp regression |
| Cell 3 (Cand 2 city boost) | gated on Cell 1 city-rate analysis | not started |
| Cell 4 (Cand 1 + 3a/3b stacked on Cell 1) | deferred | not started |
| Cand 8 vs Cand 10 ablation | open question 2 in Cell 1 journal | not started |

## Memory items to update

- The "Cell 2 next: Cand 7" standing instruction is **discharged**. Cand 7 rolled back.
- New learning to record: `val_top1` is not a reliable signal under any of the loss-aug interventions tried (Cand 8/10, Cand 7, Cand 1) — always wait for the tournament. This is the second time we've been misled by it.
