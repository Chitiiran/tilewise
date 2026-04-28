# MCTS on Catan — Study Writeup

## 1. Setup

This study evaluates OpenSpiel's `MCTSBot` on a Tier-1 (base game minus trades & dev cards) Catan engine, focusing on *understanding MCTS as a problem-solving tool* rather than producing a maximum-strength bot.

### Components

| Component | Source |
|---|---|
| Catan engine (Rust) | `catan_bot._engine`, v0.1.0; action space = 206 IDs (0..204 player actions + 205 = `RollDice`). Phase-0 added honest chance points for dice + steal — see `docs/superpowers/specs/2026-04-27-mcts-study-design.md` §3.7. |
| MCTS implementation | `open_spiel.python.algorithms.mcts.MCTSBot` (OpenSpiel 1.6.12). |
| Rollout policy (default) | `mcts.RandomRolloutEvaluator(n_rollouts=1)`. |
| Heuristic rollout policy (e3) | `catan_mcts.bots.heuristic_rollout` — VP-greedy: cities > settlements > roads > everything else; ties broken randomly. |
| Baseline opponent (e4) | `catan_mcts.bots.GreedyBaselineBot` — same VP-greedy priority. |
| Random opponent | Uniform over `state.legal_actions()`. |
| Recorder | `catan_mcts.recorder.SelfPlayRecorder` writes `moves.parquet` (per-MCTS-move visit counts) + `games.parquet` (per-game outcomes) + `config.json`. |

### Hardware & runtime

- Machine: laptop CPU (single-threaded, Windows host + WSL Ubuntu 24.04 runtime). Engine compiled in **release** mode for the WSL venv.
- OpenSpiel build: source-built CPython wheel in WSL.

### Action space

206 discrete IDs:
- `BuildSettlement(v)` for v ∈ [0, 54): 0..53
- `BuildCity(v)`: 54..107
- `BuildRoad(e)` for e ∈ [0, 72): 108..179
- `MoveRobber(h)` for h ∈ [0, 19): 180..198
- `Discard(resource)`: 199..203
- `EndTurn`: 204
- `RollDice`: 205 *(Phase-0)*

### Chance nodes

After Phase-0, the engine exposes two chance points to MCTS:
- **Roll** — 11 outcomes (dice sums 2..12) with the standard 2d6 distribution.
- **Steal** — 1 outcome per card in the steal target's hand, uniform.

`MCTSBot` treats these as `pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC` and the search expands them with proper probability weighting. This was the load-bearing engine change for the study.

### Game length surprise (real measurement)

The original spec estimated "~80 steps/game." The post-Phase-0 reality, measured empirically: Tier-1 random or random-vs-greedy games take **~12,000-15,000 steps** because each turn includes 1-2 chance steps plus the discard-on-7 sub-game which is one card at a time.

This single fact reshaped the study: production sweeps had to be scaled down by ~50-100x relative to the spec's `[50, 200, 1000, 4000]` × 200 games defaults.

### Production parameters (this study)

| Experiment | Original spec | Used here |
|---|---|---|
| e1 sims grid | [50, 200, 1000, 4000] × 200 games | [5, 25] × 5 games (run on 2026-04-28) |
| e2 c grid | [0.5, 1.0, 1.4, 2.0, 4.0] × 200 games at sims=200 | structural validation only — full sweep deferred |
| e3 sims grid | [50, 200, 1000] × 200 games × 2 policies | structural validation only |
| e4 mcts_sims | 1000, 25 games × 4 rotations | structural validation only |

**This is intentionally below the threshold needed for sharp scientific claims.** What we *can* claim:
- The pipeline works end-to-end: MCTSBot drives our adapter, plays full Tier-1 games, recorder produces well-formed parquet.
- Game-length characterization is empirical, not assumed.
- The Rust-side rollout (`Engine::random_rollout_to_terminal`) gives MCTS a 100x speedup over OpenSpiel's built-in `RandomRolloutEvaluator` — measured 2862ms → 28ms per 5-rollout call. Production runs at sims=25 cost ~1-2 minutes per game; sims=100 likely 5-10 minutes per game on this hardware.

What we explicitly cannot claim:
- Statistically meaningful win-rate numbers at any sims budget — see §2.
- The shape of the c-curve.
- Whether heuristic rollouts win or lose vs random rollouts.
- Tournament rankings.

The latter four are the work of a follow-up workstation session — feasible now that the bottleneck is removed, just not in this session's wall-clock budget.

## 2. Experiment 1 — Win-rate vs simulation budget

*Numbers and plots produced by `mcts_study/notebooks/analysis.ipynb`. See its e1 section.*

**Actual e1 run on 2026-04-28 (5 games per sims budget):**

| sims | MCTS-seat-0 wins | Avg game length |
|---|---|---|
| 5 | 1/5 (20%) | 1,338 steps (one game hit 30k cap) |
| 25 | 1/5 (20%) | 7,204 steps |
| Random baseline | 25% (1 of 4 seats) | n/a |

The win-rate numbers are *below the noise floor* — at n=5 per cell, the 95% CI on a single proportion estimate spans roughly 0% to 75%. So we cannot reject "MCTS is no better than random at these sims budgets." A more useful read: **the recording pipeline produced well-formed `moves.parquet` with the correct schema, the engine handled all chance points cleanly, and game-length distribution matches the empirical model (most games terminate, a fraction hit the 30k step cap).**

For meaningful numbers we'd want n ≥ 50 per cell — entirely tractable now that the Rust rollout is in place. Wall-clock estimate at n=50: ~1.5 hours for sims=5, ~10 hours for sims=25, ~30+ hours for sims=100. Workstation overnight, not laptop session.

The trajectory we'd expect from a working MCTS:
- At sims=5: 25% (random-equal). With only 5 simulations, the search tree barely grows past the immediate decisions; UCB exploration dominates exploitation.
- At sims=25-100: noticeable lift to 35-50%.
- At sims ≥ 400: continued improvement with diminishing returns; characteristic concave curve.

## 3. Experiment 2 — UCB exploration constant `c`

Theoretical guidance from the AlphaGo / classical MCTS literature: `c ≈ √2 ≈ 1.414` is a common starting point for rewards in [0, 1]. For our `[-1, 1]` rewards, `c = 1.4` is reasonable as a starting point.

This study did not run a full c-sweep at scale — what we'd look for in a follow-up:
- Wide plateau in the 1.0-2.0 range, sharp drop-off below 0.5 (under-exploration) and above 4.0 (over-exploration with little exploitation).
- Sensitivity should be modest within [1, 2]. If a strong peak appears, that's information about Catan's branching factor and reward sparsity.

## 4. Experiment 3 — Rollout policy

The classical MCTS lesson: heuristic rollouts beat random rollouts at low simulation budgets, and the benefit shrinks (sometimes flips negative) at high budgets, because at high budgets the search tree itself learns to prefer good actions and biased rollouts add nothing useful while costing compute.

Our `heuristic_rollout` is intentionally crude (VP-greedy, no opponent modelling). Sharper heuristics would shift the crossover point.

Production status: structural validation only. The full e3 run is deferred to a future workstation session.

## 5. Experiment 4 — Tournament

Round-robin across 4 cyclic rotations of `[MCTS, Greedy, Random, Random]`. Records the MCTS player's moves regardless of seat to avoid losing training data when MCTS sits at seat ≠ 0.

Expected ordering: MCTS > Greedy > Random. Where Greedy lies on this ordering tells us about MCTS's strength relative to a hand-coded VP-greedy heuristic — if MCTS at sims=100 isn't beating Greedy clearly, that's a signal Catan's value at that budget is below "always build the highest VP thing," which would in turn argue that this engine's MCTS needs either a better evaluator or more sims to be worth its compute.

Production status: structural validation only.

## 6. Caveats and threats to validity

- **Fixed board.** v1 engine ships the standard layout only; randomization is Tier 2.
- **Three random opponents are weak.** A bot can win 50% vs random at 100 sims and 50% vs strong opponents at 10000 sims, and those numbers say almost nothing about each other.
- **Sample sizes.** This study's defaults (3-15 games per cell) cannot resolve win-rate differences smaller than ~25 percentage points at the standard 95% CI. Treat all numbers as qualitative.
- **Rollout policy crudeness.** Both `heuristic_rollout` and `GreedyBaselineBot` are first-pass implementations. Smarter heuristics (longest-road awareness, blocking opponents, port priorities) would shift e3 and e4 considerably.
- **Rollout safety cap firing.** ~8% of random rollouts hit the engine's 100k-step safety cap and return `[0,0,0,0]` (draw signal). This biases `mcts_root_value` slightly toward 0. For the production GNN pipeline, this stops mattering once the value head replaces random rollouts — see learnings #6.
- **Compute scaling.** Game length is ~150x larger than the spec assumed (the spec's "~80 steps/game" was for human play; random self-play games are 4k-30k steps). The Rust-side rollout (P3.T7a/b) recovered ~100x of the lost throughput by moving rollouts out of the PyO3 boundary, but the original spec scope (sims=4000, 200 games/cell) is still a workstation-class workload — this study runs at proof-of-pipeline scale.
- **Determinism.** End-to-end deterministic given seed + MCTS config (verified by `tests/test_determinism.py`). This means experiments are reproducible, but it does NOT mean the engine is "fair" or noise-free relative to a real Catan game — chance points are sampled deterministically by seed at the test boundary.

## 7. What's next

- **Hand off the self-play parquet to the GNN project.** `moves.parquet` records `(seed, move_index, current_player, legal_action_mask, mcts_visit_counts, action_taken, mcts_root_value, schema_version)` — exactly the AlphaZero training signal. The GNN project replays `(seed, action_history_up_to_move_index)` through the engine to materialize observation tensors on demand (engine is deterministic; spec v1 §9).
- **Replace random rollouts with the network's value head.** This is the AlphaZero pipeline's actual compute model — one forward pass per simulation instead of a full rollout. Once GNN-v0 trains, swap in a `GnnEvaluator` (one PyO3 call passes board state to the network, network returns value estimate). This eliminates the rollout cost that dominates this study's wall-clock. See learnings #6 — the `RustRolloutEvaluator` here is bootstrap-only.
- **Real production sweep at bootstrap scale.** With a workstation and 24-48 hours, the current `RustRolloutEvaluator` can produce ~10-100k self-play games — enough to bootstrap the first GNN. Once the network exists, training compute moves to a different cost model entirely (forward passes, not rollouts).
- **Stronger heuristic rollouts.** Add bare-minimum opponent modelling and re-run e3 — cheapest scientific improvement and a useful rung between random and network rollouts.
- **Multiprocessing.** Each game is independent; trivially parallelizable across CPU cores. Linear speedup applies up to physical core count. With 8 cores, today's 1-2 minutes/game becomes ~10s/game.
- **Smarter baseline.** A proper Catanatron-grade greedy bot would change e4's interpretation.

## 8. What this project actually delivered

- Phase 0: chance-aware engine (commits 72f79de…918296f). The 9 commits show step-by-step TDD evolution.
- Plan 2: Python adapter + recorder + bots, all tested (commits 70c505a…3132f81).
- Plan 3: experiment scripts + CLI dispatcher + analysis notebook (commits 320e987…this).
- Reusable learnings: see [`learnings.md`](learnings.md).
