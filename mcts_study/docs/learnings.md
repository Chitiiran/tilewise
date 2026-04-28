# Reusable learnings from the MCTS study

Each entry below is something portable to a different MCTS application — concrete, surprising, and earned by doing the work, not just rephrasing textbook knowledge.

## 1. Honest chance nodes are non-negotiable on stochastic games

When the engine doesn't separate "the player's decision" from "the environment's roll," MCTS treats dice as just another action. The tree branches on dice outcomes as if the agent chose them, and UCB statistics measure noise rather than expected return — meaning sweeps over `c` and rollout-policy comparisons become useless.

The Phase-0 refactor of the catan_bot engine made dice and robber-steal explicit chance points. This was a substantial engine change (introduced `Action::RollDice`, promoted `GamePhase::Steal` from dead code to a live phase, exposed `is_chance_pending` / `chance_outcomes` / `apply_chance_outcome` through PyO3) — but it was the work that made everything downstream meaningful. Done the other way (faking chance), the experiments would have produced numbers but no science.

**Carry-over rule:** if you're applying MCTS or any tree-search method to a stochastic game, the FIRST audit is on whether the engine cleanly separates chance from agent decisions. Spend the engine refactor budget there before tuning anything in the search.

## 2. Game length is empirical, not derivable from rules

The MCTS-study design spec estimated "~80 steps/game" for Tier-1 Catan. Empirical reality is **~12,000-15,000 steps** — a 150x miss. The cause: chance nodes count as "steps," and so do every-card-discarded transitions on a 7. Each rolled 7 with hands > 7 produces 4-12 discrete discard steps, and there are several 7s per game.

This shaped every Plan-3 sizing decision. The spec's defaults (`sims=4000, num_games=200/cell`) were workstation-decade workloads at actual game lengths, not the laptop-hours the spec implied.

**Carry-over rule:** before sizing experiments, run *one* baseline game to terminal under the actual policy you'll use, and measure. Don't trust the spec's step-count math — game-length intuition is wrong by a constant factor whose sign is "more than you think."

## 3. Smoke tests have to be calibrated to the wall-clock budget

A "smoke test" that takes 30 minutes isn't a smoke test; it's a CI killer. The MCTSBot smoke test (P2.T6) initially ran 50 sims/move × full-rollout-to-terminal — that's roughly 50 × 12k = 600,000 random-policy state ops per MCTS move, ~1000 MCTS moves per game = 6×10⁸ ops. Took >5 minutes in release mode under WSL.

The fix: lower `max_simulations` to a value that *exercises* the search code path without paying for *strength*. We landed on `sims=5` for the MCTSBot smoke and `sims=2` for the per-experiment smokes — small enough that smokes finish in 1-2 minutes, large enough that the search tree visits at least one child action per move.

The structural-vs-slow split in `tests/test_e4_runs.py` (using `pytest.mark.slow`) is another expression of this: routine pytest runs validate the contract; opt-in slow runs validate the full pipeline.

**Carry-over rule:** smoke tests prove "this code can run" not "this code performs well." Production parameters belong in a separate harness invoked by `python -m project run all` (or equivalent), not in the test suite.

## 4. The visit-count distribution is the policy training target — get it out cleanly

For AlphaZero-style training, the natural target for the policy network is *not* the action MCTS picks (that's noisy at low sims) but the **distribution of visit counts** at the root. Picking up `mcts.SearchNode.children` per move and converting to a fixed-width array (`visit_counts_from_root` in `recorder.py`) was the smallest helper that mattered most: it's the only path from MCTS internals to the GNN-project training data.

Storing the action and not the distribution would have meant retraining MCTS to regenerate it later. Storing the distribution at recording time means the GNN project becomes a pure consumer of `moves.parquet` — replay seeds to materialize observations, mix with stored visit counts and outcomes, train. The whole shape of the AlphaZero pipeline lives or dies on this.

**Carry-over rule:** when the MCTS run is the most expensive step in your pipeline, capture *every* artifact you might want downstream — visit counts, root values, search depths — even if you're not sure you'll use them. Storage is cheap; re-running MCTS is not.

## 5. Engine + venv environment is harder than the algorithm

The Plan-2 work that drained the most session time was *not* the OpenSpiel adapter or the recorder — it was figuring out that OpenSpiel doesn't have Windows wheels, that source builds need clang+make+cmake, and that maturin + PyO3 on Linux requires a separately-built `.so` from the Windows-built `.pyd`. The actual algorithm code was relatively quick to write once these were in place.

WSL with a Linux home-fs venv (~/catan_mcts_venvs/mcts-study/) was the pragmatic answer; the research team that ships OpenSpiel admits Windows isn't supported.

**Carry-over rule:** when picking a third-party algorithm library, check Windows wheel availability *before* committing to it. If there isn't one, decide upfront whether WSL or a different library wins. Don't discover this in the middle of plan execution.

## 6. Random rollouts are a bootstrap, not the production evaluator

The slowness of MCTS on Catan is dominated by **random rollouts**: each MCTS simulation rolls a uniformly-random policy to terminal. With realistic chance sampling (post-Phase-0), random self-play games take 4,000-30,000 steps on Tier-1 — sometimes hitting a 100k-step safety cap because nobody is steering toward 10 VP.

This shapes a tempting (wrong) conclusion: "MCTS is too slow for AlphaZero-style training on this engine." That conclusion misreads what the *production* AlphaZero pipeline does:

- AlphaZero replaces random rollouts with the **value head** of the trained network. One forward pass per simulation, not a full rollout. ~1 ms on GPU.
- Per-game cost goes from "thousands of state operations × hundreds of sims" to "hundreds of forward passes × hundreds of sims."
- Game length under MCTS-with-network-evaluator drops dramatically because both sides play purposefully toward 10 VP. Empirical Catan numbers from human play are 50-150 turns; that's roughly the v1 spec's "~80 steps" estimate, which was wrong about *random* play but right about *real* play.

So this study's `RustRolloutEvaluator` is **bootstrap-quality**: it generates the first batch of training data so the GNN has something to learn from. After GNN-v0 trains, a `GnnEvaluator` swaps in. The rollout-cost bottleneck stops mattering.

**Carry-over rule:** when measuring an MCTS pipeline's compute cost, ask which phase the cost lives in. If it's the rollout phase, that cost is paid once during bootstrap and stops mattering after the first network. If it's the search phase, that's the AlphaZero per-step cost and it stays. Don't conflate the two.

---

## Three things I'd do differently next time

1. **Run one full baseline game before writing any plan.** Both the original spec's 80-step estimate and the original sims-grid defaults would have been caught immediately by 30 seconds of measurement. The plan-writing process didn't include this and paid for it across many tasks.
2. **Build engines in release mode by default for benchmark-y tasks.** The first MCTSBot smoke test ran under debug-mode Rust because the venv-side `maturin develop` defaulted to dev profile. Switching to `--release` was the difference between "smoke test takes 5 minutes" and "smoke test takes 1 minute" — which over an experiment harness is the difference between "iterate on it" and "give up on it."
3. **Reserve a session for the python-toolchain bootstrap separately from the algorithm work.** Mixing "install OpenSpiel on this OS" and "implement the OpenSpiel adapter" in the same flow burned a lot of context that could have been split cleanly.
