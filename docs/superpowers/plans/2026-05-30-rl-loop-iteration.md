# One AlphaZero-style RL loop iteration — autonomous execution plan

**Date:** 2026-05-30 (start ~22:35 EDT), deadline ~06:00 EDT 2026-05-31 (~7.5h budget).
**Trigger:** User directive — "make decision to finish a loop of RL, commit as you move through milestones, keep going autonomously until 6am 31st May."

## Goal

Complete ONE full RL iteration on the corrected async self-play stack:
**self-play(Cell 6) → train(warm-start Cell 6) → evaluate(new vs Cell 6).**
Answer: does one AlphaZero-style iteration improve over the parent net under full-Catan rules?

This is the payoff of the batched-evaluator work: the async stack has the value-perspective fix [[project_gnn_value_perspective_bug_2026_05_30]], so the self-play value targets and MCTS are now CORRECT (unlike the old buggy evaluator).

## Decisions (made autonomously, recorded here)

1. **Gate 2 (the e10e clean re-run) is DEFERRED.** Its harness is built, committed (`34f039a`), and spec-cleared (9/9 checks). But the user reprioritized to the RL loop, and both want the single GPU. RL wins. Gate 2 can run later.
2. **Warm-start from Cell 6** (`06_cand11_cand8_cand10_h128_l4` epoch10) — the current full-Catan cumulative best [[project_cell6_fullcatan_winner_2026_05_27.md]]. This is the AlphaZero "previous net" the new one improves on.
3. **Self-play sims=100** (not 200): for data generation, half the per-game cost (~30s/game vs 59s) buys ~2× more games in the budget; sims=100 still gives strong visit-count policy targets. Standard for early AZ iterations.
4. **Time-boxed self-play**, not fixed game count: run with a wall-clock cap (`max_seconds`) so it stops cleanly within budget regardless of exact per-game time. Per-game persistence means all finished games are saved.
5. **Full Catan (vp=10, bonuses)** — the production target distribution (the lesson from the rule-conditional matrix: train in the target distribution).
6. **Training:** `train.py --init-from <cell6> --run-dirs <selfplay> --rotate` (4× perspective aug), GPU, a few epochs. Warm-start means few epochs suffice.
7. **Eval:** head-to-head new-vs-Cell6 (+ 2 fillers) full Catan, modest game count (~80-120) for a directional answer within budget.

## Budget sequencing (~7.5h)

| Phase | Window (EDT) | What |
|---|---|---|
| Plan + launch self-play | 22:35–22:55 | this doc; launch `self_play_async` background |
| Self-play data-gen | 22:55–02:00 (~3h) | ~300-360 games, full Catan, sims=100, Cell 6 |
| Train | 02:00–03:45 | warm-start Cell 6, rotate aug, GPU, N epochs |
| Evaluate | 03:45–05:15 | new vs Cell 6 head-to-head, full Catan |
| Journal + commit + wrap | 05:15–06:00 | results journal, memory, branch finish |

Checkpoints via ScheduleWakeup; commit at each milestone. If a phase overruns, truncate the next (e.g. fewer eval games) to land a COMPLETE loop by 06:00 rather than a half-finished phase.

## Risks + mitigations

- **Self-play slower than estimated** → time-box catches it; train on whatever games landed (even 150 games × ~1500 decisions = 225k positions is plenty).
- **WSL/GPU flake** → per-game persistence + resume; `wsl --shutdown` recovery known.
- **Training NaN / no improvement** → warm-start + low LR; if new net regresses, that's still a VALID loop result to journal (one iteration isn't guaranteed to improve; the infra working is the deliverable).
- **Value-head still needs many iterations** → one iteration may not beat the parent; the goal is a COMPLETE, CORRECT loop, not guaranteed SOTA.

## Success criteria

A complete, committed RL iteration: self-play data on disk, a trained checkpoint, a head-to-head result, and a journal stating whether it improved — by 06:00 EDT. The loop CLOSING is the deliverable; winrate is the finding.
