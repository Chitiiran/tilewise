# v3.1 Journal — Engine flags + game-length baseline

**Date:** 2026-05-02
**Branch:** `v3` @ post-7082d83
**Tasks:** v3.0 (worktree setup) + v3.1 (engine flags, tests, baseline measurement)

## What changed

### `Engine::with_rules(seed, vp_target, bonuses_enabled)`

Two reversible runtime flags on the constructor:
- `vp_target: u8` — game-ending VP threshold. v2 default = `WIN_VP` = 10. v3 = 5.
- `bonuses_enabled: bool` — when `false`, longest-road and largest-army still **track** (length, holder, knights count) but do NOT award the +2 VP. v2 default = `true`. v3 = `false`.

`Engine::new(seed)` is unchanged in behavior — it now delegates to `Engine::with_rules(seed, WIN_VP, true)`. So every existing v2 call site stays bit-exact.

### Code paths touched

- `state.rs`: added `vp_target: u8` and `bonuses_enabled: bool` to `GameState`. Initialized to `(WIN_VP, true)` in `GameState::new` so the v2 contract holds when constructing state directly.
- `engine.rs`: added `Engine::with_rules`. `Engine::new` now delegates.
- `rules.rs`:
  - `check_win` reads `state.vp_target` instead of `crate::state::WIN_VP`.
  - `update_longest_road` and `update_largest_army`: gated all four `state.vp[p] += 2` and `state.vp[p] = state.vp[p].saturating_sub(2)` lines on `state.bonuses_enabled`. Length/holder/knights bookkeeping continues unchanged regardless of the flag.
  - Added test-only helpers `check_win_for_test` and `transfer_longest_road_holder_for_test` to drive specific code paths in `tests/v3_rules.rs` without needing real road edges.

### New test file: `tests/v3_rules.rs`

9 tests covering:
- `Engine::new` defaults match v2 (vp_target=10, bonuses_enabled=true).
- `with_rules` stores both flags faithfully.
- `vp_target` actually controls when `check_win` fires (4 vs 5 vs 9 vs 10 cases).
- LR/LA grant +2 VP iff `bonuses_enabled=true`.
- LR/LA holder transfers still update `longest_road_holder` / `largest_army_holder` even when bonuses are off.
- `saturating_sub(2)` is safe on a state where the +2 was never granted.

### v3_baseline binary

`cargo run --release --bin v3_baseline` drives 200 random-play games per rule set and emits one JSON object per rule set.

Output saved to `mcts_study/runs/benchmarks/v3_baseline.jsonl`.

## Measurements (200 games, uniform-random play, seeds 0..199)

| Metric | v2 (10 VP, bonuses) | v3 (5 VP, no bonuses) | v3 / v2 |
|---|---|---|---|
| Median length (steps) | 997 | 570 | 0.57 |
| Mean length (steps) | 1024 | 591 | 0.58 |
| P75 length | 1193 | 728 | 0.61 |
| Max length | 2910 | 1449 | 0.50 |
| ms / game | 1.2 | 0.7 | 0.58 |
| Games / sec | 820 | 1523 | 1.86 |
| Mean winner VP | 10.05 | 5.00 | — |
| Timeouts (cap=30k) | 0 | 0 | — |
| Cap-hit rate | 0% | 0% | — |

### Reading the numbers

**v3 games are ~1.7× faster on the Rust hot path.** Under random play with 5 VP and no LR/LA the median game ends in 570 steps vs 997 for v2 — close to a 1:2 ratio of VP target ratios, slightly worse than that because the early-game "build infrastructure" cost is constant across rule sets.

**Mean winner VP confirms the flag is working end-to-end.** v2 = 10.05 (sometimes a player crosses 10 with a final +2 LR/LA bonus). v3 = exactly 5.00 (no possibility of overshooting since LR/LA grants no VP).

**Distribution shape carries over.** P25/P75 ratios are similar between the two rule sets, suggesting v3 games aren't qualitatively different — just shorter. That's the desired behavior for the "throughput" hypothesis.

**Wall-clock per game is now 0.7ms.** At this rate, the v3.4 validation sweep's 1k random games would take ~0.7 seconds in pure Rust. The actual lookahead-MCTS data-gen sweep will be CPU-bound on Python+PyO3 sim work, not on game length, so the per-game speedup is mostly upper-bounded by the sim budget reduction (which lookahead-MCTS gets for free as games shorten).

### Open question — cap_hits

Both rule sets report 0 cap-hits at 30k steps. Note: the bench_random_game profile from v2 reported cap-hits of ~8% at the older 100k cap; this binary uses a 30k cap and got 0/200 at 30k under uniform-random play. v3 should be even less prone since games are shorter. We can probably leave the cap as-is for v3.

## What's next

- v3.2: PyO3 surface + Python adapter wiring so the Python side can call `Engine::with_rules`.
- v3.3: `LookaheadMctsV3` player class with the per-player VP-aware exponential sim schedule (`max(50, round(200 * 0.7^acting_vp))`).
- v3.4: 1k-game validation sweep with `LookaheadMctsV3` self-play, train 5 epochs, run e9 tournament.

## Time spent

- v3.0 (worktree + cleanup): ~10 min
- v3.1 engine flags + tests: ~25 min
- v3.1 baseline binary + measurement: ~15 min

**Total: ~50 min.** Below the spec's implicit budget — TDD made the engine work fast since I had concrete test cases driving the diff.
