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
| e1 sims grid | [50, 200, 1000, 4000] × 200 games | [5, 25, 100] × 3 games (proof-of-pipeline scale) |
| e2 c grid | [0.5, 1.0, 1.4, 2.0, 4.0] × 200 games at sims=200 | structural validation only — full sweep deferred |
| e3 sims grid | [50, 200, 1000] × 200 games × 2 policies | structural validation only |
| e4 mcts_sims | 1000, 25 games × 4 rotations | structural validation only |

**This is intentionally below the threshold needed for sharp scientific claims.** What we *can* claim:
- The pipeline works end-to-end: MCTSBot drives our adapter, plays full Tier-1 games, recorder produces well-formed parquet.
- Game-length characterization is empirical, not assumed.
- The sims-vs-strength trend can be inspected qualitatively from the small sweeps.

What we explicitly cannot claim:
- Statistically meaningful win-rate numbers at any sims budget.
- The shape of the c-curve.
- Whether heuristic rollouts win or lose vs random rollouts.
- Tournament rankings.

The latter four are the work of a follow-up study with realistic compute (a workstation, ideally GPU-vectorized, or just patience with overnight runs).

## 2. Experiment 1 — Win-rate vs simulation budget

*Numbers and plots produced by `mcts_study/notebooks/analysis.ipynb`. See its e1 section.*

The trajectory expected from a successful MCTS implementation on this engine:
- At sims=5: barely better than random (the rollout-tree is tiny, decisions dominated by exploration).
- At sims=25-100: noticeable lift over the 25% random-equal-share baseline.
- At sims ≥ 400: continued improvement but diminishing returns; characteristic concave curve.

What the actual 3-game-per-cell run shows: see notebook output. Caveat: 3 games is well below noise floor for a 4-player win-rate.

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
- **Compute scaling.** Game length is ~150x larger than the spec assumed. Plan-3's "production" scope was downsized accordingly; the original intent (sims=4000, 200 games/cell) is a workstation-class workload.
- **Determinism.** End-to-end deterministic given seed + MCTS config (verified by `tests/test_determinism.py`). This means experiments are reproducible, but it does NOT mean the engine is "fair" or noise-free relative to a real Catan game — chance points are sampled deterministically by seed at the test boundary.

## 7. What's next

- **Hand off the self-play parquet to the GNN project.** `moves.parquet` records `(seed, move_index, current_player, legal_action_mask, mcts_visit_counts, action_taken, mcts_root_value, schema_version)` — exactly the AlphaZero training signal. The GNN project replays `(seed, action_history_up_to_move_index)` through the engine to materialize observation tensors on demand (engine is deterministic; spec v1 §9).
- **Real production sweep.** With a workstation and 24-48 hours: run the spec's original sims grids and game counts. The data files this writeup wraps around are already structured for that scale.
- **Stronger heuristic rollouts.** Add bare-minimum opponent modelling and re-run e3 — this is the cheapest scientific improvement.
- **Multiprocessing.** Each game is independent; trivially parallelizable across CPU cores. Linear speedup applies up to physical core count.
- **Smarter baseline.** A proper Catanatron-grade greedy bot would change e4's interpretation.

## 8. What this project actually delivered

- Phase 0: chance-aware engine (commits 72f79de…918296f). The 9 commits show step-by-step TDD evolution.
- Plan 2: Python adapter + recorder + bots, all tested (commits 70c505a…3132f81).
- Plan 3: experiment scripts + CLI dispatcher + analysis notebook (commits 320e987…this).
- Reusable learnings: see [`learnings.md`](learnings.md).
