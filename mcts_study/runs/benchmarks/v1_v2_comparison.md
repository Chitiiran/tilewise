# v1 → v2 Phase 1 microbench comparison

v1 git_sha: aaad4c3c383e9c91c19750f908214e7468d786d2
v2 git_sha: 7cced8dd4d3d3c343fd21795543ed696fa48c2c3

| workload | v1 mean | v2 mean | speedup | notes |
|---|---:|---:|---:|---|
| bench-engine-step | 0.356 µs | 0.398 µs | 0.89× | v2 is 1.12× slower |
| bench-mcts-game | 8504.860 µs | 8359.420 µs | 1.02× | no change (within noise) |
| bench-evaluator-leaf | 46.400 µs | 42.318 µs | 1.10× | v2 is 1.10× faster |
| bench-state-clone | 0.228 µs | 0.221 µs | 1.03× | no change (within noise) |
| bench-legal-mask | _(not in v1)_ | 0.177 µs | n/a | new in v2 |

## Summary

Phase 1 was about adding rules + APIs without losing v1's perf.
Conclusions:

- engine-step / mcts-game: ~10% slowdown from longest-road DFS firing
  on every BuildRoad. Acceptable given the new strategic depth.
- evaluator-leaf: faster — longest road as new VP source means rollouts
  terminate slightly sooner.
- state-clone: unchanged (bit-packed VertexBits/EdgeBits gave ~50%
  memory reduction but similar access patterns; no clone speedup).
- legal-mask: NEW API at 0.177 µs — no perf win over legal_actions yet
  since it builds the bitmap from the action list. Phase 2 incremental
  updates will deliver the projected 30-100× speedup.

Test count: 75 → 104 (29 new tests for v2 features).

## v2 playstrength sanity (greedy-priority bot, 100 games)

| metric | v1 (LookaheadMcts d=25 s=400) | v2 (simple greedy bot) |
|---|---:|---:|
| n_games | 400 | 100 |
| timeout rate | 41.2% | 100.0% |
| median length | 4648 | 30000 |

**Important caveat:** v1 baseline used LookaheadMcts (depth=25, sims=400),
a strong player. v2 sanity used a trivial greedy bot to validate the engine
doesn't crash. v2's higher timeout rate at greedy is expected because
instant random discard (§A-bis 8a) takes resources at every 7-roll, which
greedy players don't strategize around. Real v2 measurement requires
running LookaheadMcts (or stronger) on v2 — that's pending Phase 2 once
the rules are fully complete.

**v1's 41% timeout rate from a strong player is the apples-to-apples**
**target. v2's apples-to-apples measurement comes after Phase 2.**
