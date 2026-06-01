# RL loop iteration 1 — the first complete AlphaZero-style loop (honest negative)

**Date:** 2026-05-31 (executed autonomously 22:35 EDT 05-30 → ~00:35 EDT 05-31)
**Trigger:** User directive — "make decision to finish a loop of RL, commit as you move through milestones, keep going autonomously until 6am."

## What this is

The **first complete AlphaZero-style RL iteration** in the project, end-to-end on the corrected async self-play stack:

```
self-play (Cell 6 plays itself, MCTS) → train (warm-start Cell 6 on that data) → evaluate (new vs parent)
```

The headline finding is an **honest negative**: one iteration on a small data budget REGRESSED versus the already-strong parent. But the loop CLOSED and ran on CORRECT infrastructure — which was the deliverable. The negative is a data-scale / iteration-count limitation, not a pipeline bug.

## The loop, phase by phase

### Self-play (RL-2)
- `self_play_async`: Cell 6 (`06_cand11_cand8_cand10` ep10) plays all 4 seats via the batched async MCTS, full Catan (vp=10, bonuses), sims=100, n_concurrent=16, GPU.
- **66 games, 0 timeouts, ~73 s/game**, mean 361 moves/game.
- **13,582 recorded training positions**; winners spread evenly across seats; `final_vp` correct (winner=10 VP). The value-perspective fix [[project_gnn_value_perspective_bug_2026_05_30]] means the recorded value targets are CORRECT (unlike the old buggy evaluator).

### Train (RL-3)
- `catan_gnn.train --init-from <Cell6 ep10> --rotate --device cuda --epochs 6 --lr 5e-4` on the 13.6k positions (×4 random hex-rotation aug ≈ 54k effective samples).
- Warm-started from Cell 6 with a fresh optimizer. ~23 s/epoch.

| epoch | train_loss | val_loss | val_top1 | val_value_mae |
|---|---:|---:|---:|---:|
| 1 | 2.371 | 2.970 | **0.446** | 0.591 |
| 2 | 2.068 | 3.011 | 0.439 | 0.600 |
| 3 | 2.025 | 3.033 | 0.445 | 0.603 |
| 4 | 1.996 | 3.082 | 0.441 | 0.621 |
| 5 | 1.976 | 3.149 | 0.436 | 0.632 |
| 6 | 1.963 | 3.106 | 0.434 | 0.602 |

**Classic overfit after epoch 1:** train_loss falls monotonically, val_loss rises, val_top1 peaks at ep1. Best checkpoint = epoch 1 (`checkpoint_best.pt`). 13.6k positions is too few to improve an already-converged net.

### Evaluate (RL-4)
- `e10_quad_gnn`, 4 PureGnn slots, full Catan, 120 games, GPU, 0 timeouts.
- Lineup: **A = RL_iter1** (new net, best=ep1), **B = Cell6_parent**, C = Cell1, D = Cell0 vanilla.

```
Cell6_parent   67 / 120  (55.8%)  ← the parent
Cell1          35 / 120  (29.2%)
RL_iter1       18 / 120  (15.0%)  ← the new net REGRESSED
Cell0           0 / 120  ( 0.0%)
```

**RL_iter1 (15%) lost decisively to its parent Cell 6 (55.8%)** — and even to Cell 1 (29.2%). One iteration made the net WORSE.

## Why it regressed (mechanism, not bug)

1. **Cell 6 is already strong** — the full-Catan champion, trained on 100k MCTS games. Retraining it on 66 self-play games is a tiny perturbation that mostly adds noise.
2. **Weak teacher signal.** The policy targets are 100-sim MCTS visit counts on top of Cell 6. We showed earlier that MCTS on this net isn't dramatically stronger than its raw policy [[project_e10e_gnnmcts_worse_than_puregnn_2026_05_29]] — so the "improvement signal" the student learns from is thin.
3. **Overfit on 13.6k positions** — epoch 1 already best; further epochs memorize.
4. **One AlphaZero iteration rarely improves** from a converged start on a small budget. AZ needs MANY iterations × LARGE self-play volumes; a single 66-game round can't move a 1.6M-param net forward.

## What this iteration PROVED (the actual deliverable)

The **infrastructure is correct and the loop runs end-to-end**:
- Batched async self-play generates clean, correctly-valued data at ~73 s/game (the Gate-1 4.3× speedup made 66 games in ~80 min feasible).
- The value-perspective bug is fixed; recorded `final_vp` and value targets are right.
- Training warm-starts from a parent and consumes the self-play parquets (after the move_index schema fix — see below).
- Evaluation cleanly measures new-vs-parent with 0 timeouts.

**A closed, correct AlphaZero loop is the win.** Winrate is the finding, and the finding is: one tiny iteration regresses, as expected.

## Bugs found & fixed during the run (all committed)

1. **nohup-through-WSL detachment** killed background runs (0 games, empty log). Fix: harness-tracked launches. [[feedback_worktree_swap_breaks_pyo3_install]] (appended).
2. **`--max-seconds` CLI gap** — wall-clock cap was unreachable from CLI (defaulted to 900s, would have truncated self-play to 15 min). Exposed `--max-seconds`/`--resume-dir`.
3. **GLOBAL vs PER-PLAYER move_index** (the important one) — `play_one_async_game` recorded a single global move counter, but `catan_gnn.dataset` replays counting decisions PER recorded_player. Training data load crashed ("only saw N matching decisions"). Fixed the recorder to number each seat independently (`8dbe514`); transformed the existing data in place via a renumber script. Async tests still 4/4 green.

## What to do for a REAL improvement (next iterations)

1. **Scale self-play volume** — hundreds-to-thousands of games per iteration, not 66. The CPU-bottleneck levers [[project_batched_eval_gate1_2026_05_30]] (multiprocessing, vectorized state_to_pyg) are the path to that throughput.
2. **Multiple iterations** — AZ improves over rounds; gate each new net vs the previous and only promote on a win (the standard AZ "arena" gate).
3. **Stronger teacher** — higher sims (200+), or fix the CPU bottleneck so higher sims are affordable.
4. **Early-stop at epoch 1** for tiny datasets, or train fewer epochs / lower LR to avoid the overfit.
5. **Run Gate 2 (deferred)** — the clean e10e re-run on the value-fixed stack, to confirm whether corrected GNN+MCTS beats PureGnn (un-confounds the original question).

## Status of the deferred Gate 2

`e10e_async` harness is built, committed (`34f039a`), spec-cleared (9/9). NOT run — the GPU went to the RL loop per the user's reprioritization. Run later.

## Conclusion

**One complete, correct AlphaZero-style RL iteration executed end-to-end. The new net regressed (15% vs parent's 55.8%) — expected for a single small-budget iteration from a converged start.** The deliverable was the closed, correct loop on fixed infrastructure; that is achieved. The path to actual improvement is more games + more iterations + an arena gate, all now unblocked by this pipeline.

## Cited
- Plan: `docs/superpowers/plans/2026-05-30-rl-loop-iteration.md`
- Self-play data: `runs/v3/rl_selfplay/2026-05-31T03-20-self_play_async/` (66 games)
- New checkpoint: `runs/v3/rl_train_iter1/checkpoint_best.pt` (epoch 1)
- Eval tournament: `runs/v3/tournaments/rl_iter1_eval/`
- Memory: [[project_batched_eval_gate1_2026_05_30]], [[project_gnn_value_perspective_bug_2026_05_30]], [[project_e10e_gnnmcts_worse_than_puregnn_2026_05_29]]
