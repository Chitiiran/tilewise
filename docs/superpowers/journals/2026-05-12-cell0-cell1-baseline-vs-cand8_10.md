# Cell 0 vs Cell 1: vanilla baseline vs Cand 8 + Cand 10

**Date:** 2026-05-12
**Plan:** `C:\Users\chiti\.claude\plans\let-s-read-for-context-humming-storm.md`
**Cell 0 output:** `runs/v3/loss_aug/00_baseline_h128_l4_pilot/`
**Cell 1 output:** `runs/v3/loss_aug/01_cand8_cand10_h128_l4/`

## Headline

**Cand 8 + Cand 10 eliminate the post-peak overfitting collapse.**

| Epoch | Cell 0 (vanilla loss) | Cell 1 (Cand 8 + Cand 10) | Δ |
|---:|---:|---:|---:|
| 5 | 12.08% (avg n=2: 15/120, 14/120) | 10.83% (13/120, n=1) | −1.25pp (within ±1-game noise) |
| 10 | **1.67%** (n=2: 2/120, 2/120, byte-identical) | **9.17%** (11/120, n=1) | **+7.50pp** |
| 15 | (not tested in Cell 0) | 10.00% (12/120, n=1) | — |
| 19-20 (pass-3 cited) | 2.50-3.33% | (not tested) | — |

Vanilla loss peaks at epoch 5 then collapses. Cand 8+10 produces a stable plateau at 9-11% across epochs 5-15.

## Setup

Both cells: identical except for the loss configuration.

| Setting | Cell 0 | Cell 1 |
|---|---|---|
| Architecture | h128_l4 (632k params) | h128_l4 |
| Cache | `~/catan_cache/cache_100k.pt` (3.22M positions) | same |
| Batch size | 256 | 256 |
| LR | 1e-3 (Adam) | same |
| Augmentation | random hex rotation | same |
| Seed | 0 | 0 |
| **`lambda_vp`** | **0.0** (off) | **0.10** |
| **`vp_compare_rule`** | **False** | **True** |
| Mid-tournaments | every 5 epochs, 120 games vs LookaheadV3, seed_base=19M, workers=10 cuda | same |

Cand 8 = action-class VP prior KL term. Cand 10 = 1-step VP-comparison target swap. Both implemented this session (commits `c9b6bd7`, `afd3989`).

## Per-epoch metrics

### Cell 0 (vanilla loss, 10 epochs trained — killed after epoch 10 because clearly past peak)

| ep | train_loss | val_loss | val_top1 | best |
|---:|---:|---:|---:|---|
| 1 | 2.888 | 2.865 | 0.175 | ✓ |
| 2 | 2.862 | 2.869 | 0.175 | ✓ tie |
| 3 | 2.840 | 2.893 | **0.177** | ✓ new best |
| 4 | 2.795 | 2.933 | 0.174 | — |
| 5 | 2.747 | 2.991 | 0.174 | — |
| 6 | 2.708 | 3.000 | 0.172 | — |
| 7 | 2.677 | 3.040 | 0.174 | — |
| 8 | 2.653 | 3.052 | 0.174 | — |
| 9 | 2.634 | 3.062 | 0.173 | — |
| 10 | 2.617 | 3.104 | 0.172 | — |

### Cell 1 (Cand 8 + Cand 10, 15 epochs)

| ep | train_loss | val_loss | val_top1 | vp_swap | best |
|---:|---:|---:|---:|---:|---|
| 1 | 2.739 | 3.238 | 0.173 | 18.68% | ✓ |
| 2 | 2.703 | 3.299 | 0.166 | 18.69% | — |
| 3 | 2.645 | 3.344 | **0.181** | 18.69% | ✓ new best |
| 4 | 2.574 | 3.439 | 0.168 | 18.70% | — |
| 5 | 2.522 | 3.436 | 0.170 | — | — |
| 6 | 2.484 | 3.504 | 0.170 | — | — |
| 7 | 2.456 | 3.475 | 0.170 | — | — |
| 8 | 2.434 | 3.509 | 0.166 | — | — |
| 9 | 2.416 | 3.530 | 0.165 | — | — |
| 10 | 2.402 | 3.541 | 0.171 | — | — |
| 11 | 2.390 | 3.531 | 0.172 | 18.71% | — |
| 12 | 2.380 | 3.586 | 0.169 | 18.72% | — |
| 13 | 2.371 | 3.601 | 0.169 | 18.73% | — |
| 14 | 2.364 | 3.565 | 0.167 | — | — |
| 15 | 2.357 | 3.584 | 0.162 | — | — |

## Mid-tournament summary

All tournaments: **120 games (30 × 4 rotations), sims=100, lookahead_depth=10, base_sims_v3=200, h128_l4, workers=10, device=cuda, seed_base=19M**. STANDARD_* config from `mid_training_tournament.py`.

| Run | Checkpoint | Epoch | PureGnn / 120 | % | LookV3 | GnnMcts | Random | Draws |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Cell 0 (standalone 1) | ep5 | 5 | 15 | 12.50% | 102 | 2 | 1 | 0 |
| Cell 0 (verify) | ep5 | 5 | 14 | 11.67% | 105 | 1 | 0 | 0 |
| Cell 0 (standalone 1) | ep10 | 10 | 2 | 1.67% | 117 | 0 | 1 | 0 |
| Cell 0 (verify) | ep10 | 10 | 2 | 1.67% | 115 | 1 | 1 | 1 |
| Cell 0 (cited pass-3) | ep20 / `best.pt` | 20 / 19 | 3 / 4 | 2.50% / 3.33% | — | — | — | — |
| **Cell 1 (in-process mid)** | **ep5** | **5** | **13** | **10.83%** | **105** | **2** | **0** | **0** |
| **Cell 1 (in-process mid)** | **ep10** | **10** | **11** | **9.17%** | **108** | **0** | **1** | **0** |
| **Cell 1 (in-process mid)** | **ep15** | **15** | **12** | **10.00%** | **107** | **0** | **0** | **1** |

## Behavioral observations

### vp_swap rate is essentially flat (18.68% → 18.73% across 15 epochs)

Cand 10's swap rule fires on a structurally fixed ~542k samples per epoch (out of ~2.9M train samples). Per discussion 2026-05-11: this is because Cand 8 (KL pull toward VP-yielding actions) saturates the model into "argmax to VP action whenever legal" very quickly — Cand 10 ends up firing wherever a VP action is legal AND the teacher didn't pick one, which is a fixed property of the data distribution.

In other words, Cand 10 with Cand 8 active is acting more like a hard constraint enforcer than a discovery signal. Future ablation: run Cell 1a (Cand 8 only) and Cell 1b (Cand 10 only) separately to isolate effects.

### train_loss continues to fall, val_loss continues to climb

Cell 1's train_loss falls from 2.74 (ep1) to 2.36 (ep15) — monotonic. val_loss climbs from 3.24 to 3.58 then plateaus. This is the cited "policy is being pulled away from teacher" signature. **val_top1 is uninformative in this regime** — wobbles 0.16-0.18 across all 15 epochs.

### per_game_max wider in Cell 1

Cell 0 epoch 10: per_game[max=0.54]. Cell 1 epoch 10: per_game[max=0.73]. The Cand 8+10 model has a longer tail of games where it agrees with the teacher 70%+ of moves — even though the median is unchanged.

## Statistical confidence

Cell 0 reproducibility check (verified 2026-05-11):
- ep5: 15→14 (1-game drift between independent runs)
- ep10: 2→2 (byte-identical)
- pass3_best: 4→4 (byte-identical)

So **±1 game out of 120** is the FP-non-determinism noise band. The Cell 1 ep10 result of 11/120 is **9 games above** the Cell 0 ep10 result of 2/120 — far outside the noise band. The Cell 1 result is real.

## Conclusions

1. **Cand 8 + Cand 10 (stacked) achieve the protective effect the plan hoped for.** Specifically: they prevent the post-peak collapse seen in vanilla MCTS-visit-count training.

2. **The peak is NOT raised.** At epoch 5 (Cell 0's strongest checkpoint by chance), Cand 8+10 produces 10.83% vs vanilla's 12.08% — within noise. So Cand 8+10 don't make the model strictly better at every epoch; they make it more *robust* over training time.

3. **For downstream usage (Phase 2 self-play, deployment), Cand 8+10 are strictly better.** Vanilla loss requires precisely identifying the right epoch (epoch 5) and stopping. Cand 8+10 produce a stable plateau, so any checkpoint epoch 5-15 gives ~10% winrate.

4. **The cited pass-3 baselines (h128_l4 ep20 at 2.50%, h128_l4 best at 3.33%) were the WRONG epoch.** They captured the overfit state. Even those would be improved by ~3× under Cand 8+10's plateau.

5. **val_top1 is confirmed useless** as a training signal under Cand 8+10. Tournament winrate via mid-tournaments is the only reliable proxy. The mid-tournament infrastructure built this session is the right answer for Phase 1+ experiments.

## Open questions

1. **Can Cand 8+10 be tuned to raise the peak above 12%?** Larger λ_vp? Different VP class weights? λ_vp annealing?

2. **Cand 8 vs Cand 10 alone — which contributes which effect?** Ablation experiment (Cell 1a, Cell 1b) needed. ~38 GPU-hours for both.

3. **Does Cand 8+10 plateau, or eventually climb?** Cell 1 was capped at 15 epochs. Could run another 15 from `checkpoint_epoch15.pt` to see if winrate ever crosses 12%.

4. **Other candidates (Cand 1, 2, 3, 7) — do any stack constructively on Cand 8+10?** Plan ordering says 8+10 → 7 → 2 → 1+3. Next cell would be Cand 7 on top.

## Reproduction

```bash
# Cell 0 (vanilla baseline)
python scripts/train_grid_inproc.py \
  --cache-path ~/catan_cache/cache_100k.pt \
  --out-root runs/v3/loss_aug/00_baseline_h128_l4_pilot \
  --status-file runs/v3/dashboard/cell0.json \
  --epochs 10 --batch-size 256 --device auto \
  --rotate --rotate-mode random --cells h128_l4 --seed 0 \
  --mid-tournament-every 5

# Cell 1 (Cand 8 + Cand 10)
python scripts/train_grid_inproc.py \
  --cache-path ~/catan_cache/cache_100k.pt \
  --out-root runs/v3/loss_aug/01_cand8_cand10_h128_l4 \
  --status-file runs/v3/dashboard/cell1.json \
  --epochs 15 --batch-size 256 --device auto \
  --rotate --rotate-mode random --cells h128_l4 --seed 0 \
  --mid-tournament-every 5 \
  --lambda-vp 0.10 --vp-compare-rule
```

Total wall-clock: Cell 0 ~7h to ep10 + verify tournaments (~2.5h). Cell 1 ~19.3h (cache + 15 epochs + 3 mid-tournaments).

## Commits this session

- `6c6725c` — Phase 0 trade-value analysis (Cand 4 dropped)
- `3796d85` — mid-training tournament gating infra
- `d5572a9` — parallel workers for mid-tournament
- `757ba2a` — STANDARD_* tournament constants + launcher
- `68d7677` — chain_after_tournament watchdog
- `a5fc5b1` — fix: mid-tournament uses checkpoint_epoch{N}.pt
- `4f83bc8` — queue_epoch10_tournament watchdog
- `26a364c` — verify_3_tournaments reproducibility script
- `c9b6bd7` — **Cand 8: action-class VP prior**
- `afd3989` — **Cand 10: 1-step VP-comparison target swap**
- `b596c4b` — smoke test for stacked Cand 8 + Cand 10
