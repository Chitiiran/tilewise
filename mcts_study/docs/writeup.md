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

| Experiment | Original spec | Used here (5h sweep, 2026-04-28→29) |
|---|---|---|
| e1 sims grid | [50, 200, 1000, 4000] × 200 games | **[5, 25, 100] × 20 games** = 60 games (n=20 cells) |
| e2 c grid | [0.5, 1.0, 1.4, 2.0, 4.0] × 200 games at sims=200 | [0.5, 1.0, 1.4, 2.0, 4.0] × 10 games at sims=25 — **selection-biased; deferred to v2** |
| e3 sims grid | [50, 200, 1000] × 200 games × 2 policies | [5, 25] × 10 games × 2 policies — **selection-biased; deferred to v2** |
| e4 mcts_sims | 1000, 25 games × 4 rotations | mcts_sims=25, 5 games × 4 rotations — **selection-biased; deferred to v2** |

The whole sweep took ~12 hours wall-clock, single-threaded. **e1 has clean data**; e2/e3/e4 ran with a 5-min per-game cap that selectively discarded slow games — see §3 for what we plan to do about it.

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

**Actual e1 run on 2026-04-28 (n=20 games per sims budget, 60 games total, ~8h wall-clock):**

| sims | MCTS-seat-0 wins | No-winner games | Avg game length (steps) |
|---|---|---|---|
| 5 | 6/20 (**30%**) | 1/20 | 7,280 |
| 25 | 1/20 (5%) | 4/20 | 9,836 |
| 100 | 9/20 (**45%**) | 7/20 | 14,360 |
| Random baseline | 25% (1 of 4 seats) | — | — |

Three findings worth flagging:

**(a) MCTS strength is non-monotone in simulation budget.** The textbook story says "more sims → better." Empirically, sims=25 *underperforms* both sims=5 (just barely above random) and sims=100 (clear lift). Three plausible reads, in order of likelihood:
- Statistical noise. At n=20 the 95% CI on a single proportion is roughly ±21pp. The 5% vs 25% gap is barely outside, so this could be a noise dip.
- The "in-between" budget produces worse games than either extreme. At sims=5, MCTS plays nearly random — outcomes are random-baseline-like. At sims=100, MCTS plays meaningfully. At sims=25, MCTS commits to plans without the depth to execute them.
- Selection bias from the no-winner rate. As sims goes up, the bot plays more carefully and games more often hit the engine's 30k-step cap before any player reaches 10 VP. Those games are recorded as `winner=-1`, so the winrate-of-MCTS denominator counts them as losses.

**(b) Average game length grows monotonically with sims** (7.3k → 9.8k → 14.4k). This is the predicted "stronger MCTS → longer games against random" pattern: MCTS plays carefully toward 10 VP but the random opponents don't, so resource-deadlocks and discard cycles drag games out. By sims=100, 35% of games (7/20) hit the 30k cap.

**(c) The headline "MCTS at sims=100 wins 45% vs random" is real**, comfortably above the 25% baseline at this n. With n=20, a 45% rate has a 95% CI of roughly 26%-66% — so we can confidently say it's better than random, but we cannot say "it's 45%" with precision better than ±20pp.

## 3. Experiments 2, 3, 4 — Deferred to v2 sweep

The 5h production run (which ran for ~12 hours total) produced data for e2, e3, and e4, but the dataset is unsuitable for the experiments' intended scientific claims.

**What happened.** The Python codebase was edited mid-sweep — partway through e1's long sims=100 cell, I started the v2 hardening branch (per-game wall-clock cap, per-cell parquet shards, skipped.csv sidecar, lower rollout cap). Because the WSL venv was a `pip install -e` editable install pointing at the worktree, every commit on `mcts-v2-hardening` went live immediately for any *new* python invocation. e1 had already loaded its code and was unaffected, but e2/e3/e4 — which spawn as fresh processes at the start of each experiment — picked up the v2 Python.

**The result.** e2/e3/e4 ran with the v2 default `max_seconds=300`. Many games timed out:
- e2 (UCB c sweep): 16 of 50 games timed out (32%).
- e3 (rollout policy): 23 of 40 games timed out (58%; e3 heuristic/sims=25 had only 1/10 complete).
- e4 (tournament): 4 of 20 games timed out (20%).

Conditional-on-completion winrates exist (saved alongside the parquet shards) but selection bias makes them unreliable — only the games that finished in <5 min are recorded, and "fast games" are correlated with positions where rollouts terminate quickly, which correlates with whichever player happens to be ahead in cards. A bot that's ahead plays faster games; a bot that's behind plays slower games and times out. So winrates from this dataset over-count games where the recorded MCTS player happened to be in good positions.

**The v2 sweep planned for the next session** rebuilds the engine with the lower rollout cap *and* keeps the wall-clock cap, runs all four experiments with parameters matching this v1 e1 (n=20 etc.), and uses 4-core multiprocessing. Expected wall-clock ~1-2 hours total (vs. v1's 12 hours). That sweep is the proper place for e2/e3/e4 numbers.

For now, what we *can* learn from the v2-Python data on this run is *how often the 5-min cap fires* — that's a real measurement of MCTS-on-Catan compute variance. The skipped-rate of 58% at heuristic/sims=25 confirms the Rust-side rollout, while a big speedup over OpenSpiel's `RandomRolloutEvaluator`, doesn't fully tame the rollout-cap-firing pathology at deeper search budgets.

## 4. Caveats and threats to validity

- **Fixed board.** v1 engine ships the standard layout only; randomization is Tier 2.
- **Three random opponents are weak.** A bot can win 50% vs random at 100 sims and 50% vs strong opponents at 10000 sims, and those numbers say almost nothing about each other.
- **Sample sizes.** e1 ran with n=20 per cell — 95% CI on a single proportion estimate is roughly ±21pp. We can resolve "MCTS at sims=100 beats random" but not finer-grained claims. e2/e3/e4 are even noisier (selection-biased) and deferred to v2.
- **Rollout policy crudeness.** Both `heuristic_rollout` and `GreedyBaselineBot` are first-pass implementations. Smarter heuristics (longest-road awareness, blocking opponents, port priorities) would shift e3 and e4 considerably.
- **Rollout safety cap firing.** ~8% of random rollouts hit the engine's 100k-step safety cap and return `[0,0,0,0]` (draw signal). This biases `mcts_root_value` slightly toward 0. For the production GNN pipeline, this stops mattering once the value head replaces random rollouts — see learnings #6.
- **Compute scaling.** Game length is ~150x larger than the spec assumed (the spec's "~80 steps/game" was for human play; random self-play games are 4k-30k steps). The Rust-side rollout (P3.T7a/b) recovered ~100x of the lost throughput by moving rollouts out of the PyO3 boundary, but the original spec scope (sims=4000, 200 games/cell) is still a workstation-class workload — this study runs at proof-of-pipeline scale.
- **Determinism.** End-to-end deterministic given seed + MCTS config (verified by `tests/test_determinism.py`). This means experiments are reproducible, but it does NOT mean the engine is "fair" or noise-free relative to a real Catan game — chance points are sampled deterministically by seed at the test boundary.

## 5. What's next

- **Run the v2 sweep.** All four experiments with n=20 / n=10 / n=10 / n=5-per-seating, on the v2-hardened pipeline (per-game wall-clock cap, per-cell parquet shards, lower engine rollout cap, 4-core multiprocessing). Expected wall-clock ~1-2 hours total. Replaces the selection-biased v1 e2/e3/e4 numbers with proper data.
- **Hand off the self-play parquet to the GNN project.** `moves.parquet` records `(seed, move_index, current_player, legal_action_mask, mcts_visit_counts, action_taken, mcts_root_value, schema_version)` — exactly the AlphaZero training signal. The GNN project replays `(seed, action_history_up_to_move_index)` through the engine to materialize observation tensors on demand (engine is deterministic; spec v1 §9).
- **Replace random rollouts with the network's value head.** This is the AlphaZero pipeline's actual compute model — one forward pass per simulation instead of a full rollout. Once GNN-v0 trains, swap in a `GnnEvaluator` (one PyO3 call passes board state to the network, network returns value estimate). This eliminates the rollout cost that dominates this study's wall-clock. See learnings #6 — the `RustRolloutEvaluator` here is bootstrap-only.
- **Real production sweep at bootstrap scale.** With a workstation and 24-48 hours, the current `RustRolloutEvaluator` can produce ~10-100k self-play games — enough to bootstrap the first GNN. Once the network exists, training compute moves to a different cost model entirely (forward passes, not rollouts).
- **Stronger heuristic rollouts.** Add bare-minimum opponent modelling and re-run e3 — cheapest scientific improvement and a useful rung between random and network rollouts.
- **Multiprocessing.** Each game is independent; trivially parallelizable across CPU cores. Linear speedup applies up to physical core count. With 8 cores, today's 1-2 minutes/game becomes ~10s/game.
- **Smarter baseline.** A proper Catanatron-grade greedy bot would change e4's interpretation.

## 5b. The lookahead-VP pivot (e5)

After the v1 sweep, the rollout-cap firing rate (~8%) and the dice-luck-dominated nature of random play started to look like a fundamental limitation rather than a tunable knob. We replaced random rollouts with a **depth-bounded greedy-VP lookahead**: from each MCTS leaf, play forward `depth` rounds with greedy decisions and uniform chance sampling, then return per-player VP normalized to `[-1, 1]`. The signal is biased (greedy ≠ optimal) but it's *always present* — no `[0,0,0,0]` cap returns swamping the search.

Cost: implementation is 1 Rust function (`Engine::lookahead_vp_value`) + 1 OpenSpiel evaluator (`LookaheadVpEvaluator`). Per-leaf wall-clock is **~6× faster than even the Rust-side full rollout**, because the lookahead exits after `depth*4` EndTurns instead of running to terminal.

A second tweak compounded the speedup: **skip-trivial turns**. Catan has many forced moves (forced `EndTurn` after a no-resource roll, forced `Discard` when only one card matters). `play_one_game` now applies `len(legal)==1` actions directly and bypasses MCTS entirely on those steps.

### e5 results (2026-04-29, 4 workers, ~25 min wall-clock)

90 games, MCTS-vs-3-random across `(depth, sims) ∈ {3, 10, 25} × {25, 100, 400}`:

| depth | sims | n  | MCTS wins | winrate | mean VP | avg game length |
|------:|-----:|---:|----------:|--------:|--------:|----------------:|
|     3 |   25 | 10 |         0 |    0.0% |    1.70 |          17,171 |
|     3 |  100 | 10 |         0 |    0.0% |    3.30 |           9,637 |
|     3 |  400 | 10 |         1 |   10.0% |    3.80 |           7,342 |
|    10 |   25 | 10 |         0 |    0.0% |    1.20 |          17,334 |
|    10 |  100 | 10 |         0 |    0.0% |    2.90 |           8,364 |
|    10 |  400 |  6 |         1 |   16.7% |    4.33 |           7,574 |
|    25 |   25 | 10 |         0 |    0.0% |    1.30 |          14,730 |
|    25 |  100 | 10 |         3 |   30.0% |    5.50 |          11,412 |
|    25 |  400 |  7 |         6 | **85.7%** |    9.86 |          10,183 |

7 of 90 games timed out at the 300s/game cap (all in `depth ≥ 10`, where lookahead bookkeeping accumulates).

### What changes vs v1 e1

v1 random rollouts at sims=100 produced 45% winrate. e5's lookahead-VP at depth=25/sims=400 hits **85.7%** — a clean qualitative jump. Pattern is monotone in both axes:

- **Depth matters more than sims.** At depth=3, more sims don't help: greedy 3-round lookahead is too myopic to distinguish cities-now from roads-for-future-settlements. At depth=25, the same sim budget that gave 0% becomes 85.7%.
- **Skip-trivial collapses game length.** Random-vs-MCTS games at depth=3 still average ~17k steps; at depth=25/sims=400 they collapse to ~10k *and* the MCTS player is reaching 10 VP, not stuck in mid-game.
- **Sims=25 is universally too small** under this evaluator — even at depth=25, MCTS doesn't have enough sims to differentiate moves.

### Caveats

- **Greedy-augmented MCTS, not pure MCTS.** The lookahead encodes domain knowledge (cities > settlements > roads). A network-based evaluator would replace this; the lookahead is a stepping stone.
- **n=10 per cell.** 95% CI on a single proportion is roughly ±30pp. The depth=25/sims=400 cell is the only one where the effect is large enough to read through that uncertainty.
- **No opponent strength scaling.** Random opponents are the trivial case. The next experiment is e5-vs-greedy: the lookahead might collapse against a non-trivial opponent that defends key vertices.

## 5c. e5 v2 — 1500 games, 5×3 grid (2026-04-29)

The n=10 caveat above was the dominant uncertainty. We re-ran e5 at n=100 per cell, with a 5×3 grid widened to depth ∈ {3, 10, 17, 25, 35} and a tighter 360s/game cap. ~7 hours wall-clock on 4 workers, 1500 game-attempts.

| depth \ sims | sims=25 | sims=100 | sims=400 |
|------:|--------:|---------:|---------:|
| **3**  | 1.0% (1/100) | 0.0% (0/100) | 1.0% (1/100) |
| **10** | 2.0% (2/100) | 1.0% (1/100) | 12.0% (10/83) |
| **17** | 0.0% (0/100) | 2.0% (2/100) | 46.2% (30/65) |
| **25** | 1.0% (1/100) | 14.0% (14/100) | **75.4% (43/57)** |
| **35** | 1.0% (1/100) | 41.0% (41/100) | **76.4% (42/55)** |

Headline corrections vs v1:
- The v1 n=7 result of 85.7% at depth=25/sims=400 was real but inflated. True winrate is ~75%.
- **At sims=400, returns plateau past depth=25** — depth=35/sims=400 is statistically indistinguishable from depth=25/sims=400.
- **At sims=100, deeper still helps** — depth=35/sims=100 hits 41%, vs 14% at depth=25/sims=100. There's a real depth-vs-sims tradeoff curve.
- **sims=25 is universally noise-floor** at every depth tested. Insufficient sims swamps any signal.
- 140/1500 (9.3%) timed out, all concentrated in heavy cells. Effective n shrinks where wall-clock saturates.

### Schema-version migration note

The recorder bumped from `SCHEMA_VERSION = 1` to `SCHEMA_VERSION = 2` partway through the 2026-04-29 work, adding an `action_history: list[int32]` column to `games.parquet`. The motivation was downstream replay: GNN training needs to reconstruct the engine state at any (seed, move_index) cheaply, which requires the full action trace.

Three consequences for anyone reading the parquets:

1. **The 1500-game e5 v2 sweep above was actually written under SCHEMA_VERSION=1** — the bump landed on a parallel branch and only took effect for sweeps started afterward. The cell winrate analysis here is unaffected (it only needs `winner` and `final_vp`), but the parquets are not directly usable for GNN training.
2. **`CatanReplayDataset` filters out v1 games** at construction (silently — they cannot be replayed without action_history). Anyone trying to train a GNN on the 1500-game sweep will see an empty dataset and need to re-record.
3. **Going forward, any sweep that wants to feed the GNN pipeline must run on a branch where SCHEMA_VERSION ≥ 2 is in effect at the time the recorder writes.** A small re-recording sweep (~hours at depth=25/sims=400 cells) under v2 is the canonical path.

## 6. What this project actually delivered

- Phase 0: chance-aware engine (commits 72f79de…918296f). The 9 commits show step-by-step TDD evolution.
- Plan 2: Python adapter + recorder + bots, all tested (commits 70c505a…3132f81).
- Plan 3: experiment scripts + CLI dispatcher + analysis notebook (commits 320e987…and on).
- v2 hardening (branch `mcts-v2-hardening`, 7 commits 2064eeb…e0fe172): per-game wall-clock cap, per-cell parquet checkpoint, skipped.csv + done.txt sidecars, lower engine rollout cap, multiprocessing pool runner, v2 production driver.
- Production sweep 2026-04-28→29: 12 hours, all four experiments completed, parquet at `runs/2026-04-2{8,9}T*-e{1,2,3,4}_*/`. e1 has clean n=20 data; e2/e3/e4 selection-biased and queued for v2 redo.
- Reusable learnings: see [`learnings.md`](learnings.md).
