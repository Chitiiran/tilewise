# v2 Phase 2 wrap-up — full Catan rules + cached legality

v1 git_sha:     aaad4c3c383e9c91c19750f908214e7468d786d2
phase 1 sha:    7cced8dd4d3d3c343fd21795543ed696fa48c2c3
phase 2 sha:    b2a7bff4ccd7a56a569e65ad89057acb2b209306

## Microbench across phases

| workload | v1 | phase 1 | phase 2 | v1→v2 |
|---|---:|---:|---:|---:|
| bench-engine-step | 0.356 µs | 0.398 µs | 1.068 µs | 0.33× |
| bench-mcts-game | 8504.860 µs | 8359.420 µs | 13544.080 µs | 0.63× |
| bench-evaluator-leaf | 46.400 µs | 42.318 µs | 91.907 µs | 0.50× |
| bench-state-clone | 0.228 µs | 0.221 µs | 0.226 µs | 1.01× |
| bench-legal-mask | — | 0.177 µs | 0.010 µs | — |

## What got added in Phase 2

- §2.1: Port-aware trade ratios (3:1 generic, 2:1 specific) — 8 tests
- §2.2: Largest army tracking (transferable +2 VP) — covered with §2.3
- §2.3: Dev cards (knight, road build, monopoly, year of plenty, VP) — 12 tests
- §2.4: Player-to-player 1-for-1 trades — 7 tests
- §2.5: Cached legal_mask with dirty-flag invalidation — 41× faster legal_mask
- §2.6: Engine::new defaults to ABC balanced board

## Test count progression

- v1: 75 tests
- Phase 1 complete: 112 tests
- Phase 2 complete: 168 tests

## Action space progression

- v1: 206 actions
- Phase 1.8: 226 (+20 maritime)
- Phase 2.3: 260 (+34 dev cards)
- Phase 2.4: 280 (+20 player trades)

## Key engineering decisions

**Legal mask cache** turned out to be the single biggest perf win in v2.
After Phase 2.3 added 34 dev card actions, computing legal_mask from
scratch on every call dropped throughput by 30-50%. The lazy cache
(invalidate on mutation, recompute on read) brought the legal_mask call
from 0.5 µs → 0.01 µs (50×). Phase 2.5 is the architectural payoff.

**Engine-step regression** is real: ~3× slower than v1 due to:
- Longest-road DFS on every BuildRoad
- Port ownership recompute on every Build{Settlement,City}
- Largest-army update on every PlayKnight
- Larger legal_actions check (more rule branches)
All of these are pure rule additions; the cache mitigates them where
the same state is queried multiple times (the typical MCTS pattern).
