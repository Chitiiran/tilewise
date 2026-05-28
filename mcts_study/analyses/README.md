# Analyses

One-shot analysis scripts for v3 loss-augmentation experiments. These were originally `scratch_*.py` files at `mcts_study/` root, gitignored. Tracked here as part of the 2026-05-28 repo reorg (Phase 3b) — kept for reproducibility of the journal findings.

Convention: each script is self-contained (its own hardcoded `TR =` path or similar), reads from `runs/v3/...` parquets, and prints a tabular result or saves figures. Re-running requires the source tournament/training data to exist.

## Scripts

### Cand 11 (road-pip prior) development

| Script | Purpose | Journal |
|---|---|---|
| `road_pip_calibration.py` | Pre-launch calibration of Cand 11 — gate-firing rate + prior/visits entropy ratio. Used to confirm `λ_road=0.05` was safe. | `2026-05-25-cell5-road-pip-prior.md` |
| `road_pip_profile.py` | cProfile of `road_pip_prior_loss` on synthetic batches. Identified the `.item()` CUDA-sync issue. | `2026-05-25-cand11-perf-rca.md` |
| `road_pip_timing.py` | End-to-end per-batch GPU wall-clock vs vanilla. Confirmed the vectorized fix recovered perf. | `2026-05-25-cand11-perf-rca.md` |
| `cell6_timing.py` | Cell 6 (Cand 11 + Cand 8 + Cand 10) stack timing — confirmed 1.06× vanilla overhead pre-launch. | `2026-05-26-cell6-cand11-cand8-cand10-stack.md` |

### Behavioral analyses (v3 + full-Catan)

| Script | Purpose | Journal |
|---|---|---|
| `cand11_closeout_diagnostic.py` | Cell 5 v2 (Cand 11) closeout investigation — identified the "Cand 11 has city resources but plays other actions" pattern. | `2026-05-26-cell6-cand11-cand8-cand10-stack.md` |
| `midgame_actions_cell1_ep10.py` | First per-role action analysis (Cell 1 ep10 mid-tournament). Found 7.2 roads/settle ratio bottleneck. | (informally referenced) |
| `midgame_actions_e10c_1200.py` | Action analysis on the 1200-game e10c tournament. Established the per-role behavioral signatures. | `2026-05-26-cand11-headtohead-tournament.md` |
| `pip_conversion_e10c_1200.py` | Per-role starting-pip → ending-pip → win conversion. Showed Cell 5 v2 stalls at ~5-6 VP. | (referenced in full-Catan journal) |
| `fullcatan_deep_analysis.py` | Replay-based deep behavioral analysis of the full-Catan tournament. Single largest analysis pass (~25 min). Extracts: build counts, bonus economy, trade dynamics, port usage, resource specialization, robber targeting. | `2026-05-27-fullcatan-deep-behavioral-analysis.md` |

### Plotting

| Script | Purpose | Outputs |
|---|---|---|
| `fullcatan_plots.py` | Six figures for the full-Catan deep analysis (bonus holding, knights-vs-roads, game length distribution, etc.). Re-runs the deep analysis if cache miss. | `docs/superpowers/journals/figures/{bonus_holding,knights_vs_roads,game_length_dist,winrate_by_rules,resource_specialization,bonus_contribution_to_wins}.png` |
| `rules_matrix_plot.py` | The 4-quadrant rules×opponents winrate plot. Pure data, no replay needed (hardcoded numbers from journals). | `figures/rules_opponents_matrix.png`, `figures/cell_rank_by_context.png` |

## Running

All scripts hardcode `/mnt/c/dojo/catan_bot/.claude/worktrees/v3/mcts_study/runs/v3/...` paths. With the 2026-05-28 runs-data-on-WSL symlink (`mcts_study/runs/v3 → /home/chitii/catan_data/runs/v3`), these paths resolve correctly through WSL.

From inside WSL:
```bash
source ~/catan_mcts_venvs/mcts-study/bin/activate
cd /mnt/c/dojo/catan_bot/.claude/worktrees/v3/mcts_study
python analyses/<script_name>.py
```

Some scripts (e.g. `fullcatan_deep_analysis.py`) take 20-30 min to replay all 1200 games. The plotting scripts depend on the deep-analysis script running first.

## Older scratch scripts (still gitignored)

Older diagnostic scripts (~19 from the May 9-10 grid_pass100k era) are still at `mcts_study/scratch_*.py` and remain gitignored per the `scratch_*.py` rule. They were one-shot debugging tools, not curated analyses.
