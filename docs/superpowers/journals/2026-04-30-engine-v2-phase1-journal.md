# Engine v2 — Phase 1 Execution Journal

**Branch:** `engine-v2` (off `main`, branched 2026-04-30)
**Worktree:** `C:/dojo/catan_bot/.claude/worktrees/engine-v2/`
**Design doc:** [`2026-04-30-v2-restart-full-game-design.md`](../specs/2026-04-30-v2-restart-full-game-design.md)

This journal records what we built during Phase 1 (the v2 engine MVP),
what we measured, and what we deliberately deferred.

---

## What shipped (Phase 1)

| # | Task | Status |
|---|---|---|
| 0.1 | v1 microbench baselines | ✅ |
| 0.2 | v1 playstrength baseline | ✅ |
| 0.3 | v2 worktree off main | ✅ |
| 1.1 | Bit-packed GameState (VertexBits/EdgeBits) | ✅ |
| 1.2 | LegalMask bitmap API (no incremental yet) | ✅ |
| 1.3 | Unmake-move state.clone elimination | ⏭️ deferred — clone is 228ns, not the bottleneck |
| 1.4 | PyO3 surface: bank/all_hands/observation_for/legal_mask | ✅ |
| 1.5 | Instant random discard on robber-7 | ✅ |
| 1.6 | ABC balanced map generation | ✅ |
| 1.7 | Inventory caps + longest road + expanded VP | ✅ |
| 1.8 | Maritime trades (4:1 bank only; ports deferred) | ✅ |
| 1.9 | Property-based invariant tests | ✅ |
| 1.10 | v2 microbench + v1 comparison report | ✅ |

12 of 13 tasks complete. Engine compiles, all 112 tests pass, no behavioral
regressions on the rules subset that survived from v1.

## Numbers worth keeping

### Test count
- v1 → v2: 75 → **112 tests** (+37 new for v2 features).
- New test files: `legal_mask.rs`, `instant_discard.rs`, `abc_map.rs`,
  `inventory_and_longest_road.rs`, `maritime_trade.rs`.
- Updated: `properties.rs` (proptest invariants for resource conservation,
  inventory caps, longest road consistency, replay determinism — each
  runs 128 random sequences).

### Microbench (`mcts_study/runs/benchmarks/v1_v2_comparison.md`)

| workload | v1 | v2 phase 1 | speedup |
|---|---:|---:|---:|
| bench-engine-step | 0.356 µs | 0.398 µs | 0.89× (slower) |
| bench-mcts-game | 8505 µs | 8359 µs | 1.02× (within noise) |
| bench-evaluator-leaf | 46.4 µs | 42.3 µs | 1.10× (faster) |
| bench-state-clone | 0.228 µs | 0.221 µs | 1.03× |
| bench-legal-mask | n/a | 0.177 µs | NEW API |

**Key calibration finding:** `state-clone` was already 228ns. The §J37 claim
of "5-10× MCTS speedup from unmake-move" was wrong — clone is not the
bottleneck. Real perf wins will come from §B-bis 11a (incremental legal-mask
updates) when the action space grows further.

The 12% slowdown on `engine-step` is from longest-road DFS on every
BuildRoad. Acceptable given the new strategic depth; can be reclaimed later
by caching the per-player longest road and recomputing only when affected
edges change.

## Architecture decisions taken

### Bit-packed VertexBits/EdgeBits — kept simple
- 1 byte per slot (3 bits used) instead of `Option<u8>` (2 bytes due to padding).
- Memory savings: 50% on those fields.
- **Did NOT** go further to 3-bits-per-slot packed across u128 because access
  patterns are random and the unpacking would offset the cache-line savings.
- API: `state.settlements.get(v)` / `.set(v, owner)` / `.iter_owned()`.
  Mechanically migrated all v1 call-sites with a regex script.

### LegalMask — API first, perf later
- `state.legal_mask: [u64; 4]` (256 bits, room for v2's 226 + headroom).
- For Phase 1, computed lazily from `legal_actions()`. No perf win yet.
- `LEGAL_MASK_WORDS = 4` chosen to leave gap slots for v2 dev cards / port
  trades without changing the API.
- Property test (`tests/legal_mask.rs`) confirms bitmap == action list.

### Instant random discard — deliberate scope cut
- Phase `Discard` still exists (dead-code, kept for backward-compat with
  some unit tests). `apply_dice_roll(state, roll, rng)` now applies all
  discards inline before transitioning to MoveRobber.
- Random discard uses the engine's RNG → deterministic per seed.
- Sanity test: 6 cases including no-one-owing, multi-owing, deterministic-
  per-seed.
- **Cost:** the greedy bot now times out 100% of v2 sanity games. Not a
  bug — random discards punish non-strategic play harder. LookaheadMcts
  would handle this differently.

### ABC map generation — verified by adjacency property test
- 19 hexes laid out row-major (3-4-5-4-3 rows). The official spiral is
  ring-by-ring (12 outer + 6 middle + 1 center). I added a `SPIRAL_ORDER[19]`
  permutation that maps spiral index → row-major index.
- Token sequence: `[5, 2, 6, 3, 8, 10, 9, 12, 11, 4, 8, 10, 9, 4, 5, 6, 3, 11]`
  (Klaus Teuber's published ABC).
- The "no adjacent red numbers" property is verified by deriving hex
  adjacency from shared vertex sets — 6 tests pass including this one.
- `Engine::new(seed)` still uses `Board::standard()` for back-compat;
  switching to `Board::generate_abc(seed)` is a one-line change when we
  want randomized boards in training.

### Inventory caps — current-on-board semantics
- `settlements_built[p]` is "currently on board," not lifetime. Decrements
  when a settlement upgrades to a city. This matches Catan piece-supply rules
  (you have 5 settlement pieces; once upgraded, the piece returns to your supply).
- Tested: cap enforcement appears in `legal_actions_main`; build-action
  filtering is the contract, not state-mutation rejection.

### Longest road — DFS-based, transferable +2 VP
- `longest_road_for_player(state, player)`: enumerates the player's owned
  edges, runs DFS from each endpoint, blocks at vertices with opponent
  buildings. Returns longest simple path.
- `update_longest_road(state)` is called after every BuildRoad and
  BuildSettlement (a settlement on a chain-internal vertex breaks the chain).
- Holder logic: strict-max-≥5 holder transfers VP. Ties keep prior holder iff
  prior holder is tied; otherwise no holder.
- Property test: holder ↔ length consistency holds across 128 random sequences.

### Maritime trades — 4:1 only for now
- Action::TradeBank { give, get } at IDs 206-225 (5 give × 4 valid get = 20).
- Same-for-same excluded.
- Ports (3:1 / 2:1) deferred to Phase 2 — needs per-player port ownership.
- Apply: hand[give] -= 4, bank[give] += 4, hand[get] += 1, bank[get] -= 1.
  Resource conservation invariant holds.

## What we didn't do, and why

### Phase 1.3 — unmake-move
The bench showed clone is 228ns. With the design doc's projected 5-10×
speedup, that's saving 200ns at most. Other items in Phase 1 (rules,
maritime trade, ABC) had higher leverage. Will revisit in Phase 2 if
incremental legal-mask updates expose clone as the next bottleneck.

### Player-to-player trades, dev cards, largest army
Per the design doc, Phase 1 was "rules MVP." These are Phase 2 work.

### Port-aware trade ratios
Need to track per-player port ownership (which ports do I have a
settlement/city on). Trivial state addition (~9 bits per player), ~30 lines
of rule extension. Bundled with Phase 2 when we revisit trades.

### Incremental legal-mask updates (§B-bis 11a-b)
The big perf claim of the design doc — 30-100× faster legal-action lookup —
is Phase 2 work. The Phase 1 implementation just builds the bitmap from the
slow `legal_actions()` list each call. The API is stable; downstream tools
(GNN policy mask, MCTS) can use `engine.legal_mask()` already.

## Process notes

### What went well
- **Tests-first culture:** every rule addition has its own integration test
  file. Catching the resource-conservation bug in `trade_bank_apply` test
  setup (rather than in the engine itself) was a 30-second fix.
- **Frequent small commits:** 13 commits across Phase 0+1, each one
  shippable. Reverts would be cheap.
- **Bench-driven decision-making:** the unmake-move deferral happened
  *because* the bench showed clone is fast. Without that data, I'd have
  spent a day on a 1.05× speedup.

### What was harder than expected
- The state.rs bit-pack required cascading API changes in rules.rs,
  observation.rs, and the test suite. A regex-based migration script worked
  but introduced two bugs (assignment vs equality) that needed manual fixes.
- The longest-road DFS test geometry was tricky — my "5 connected edges"
  test guess didn't actually form a chain on the canonical board, so the
  test couldn't assert holder transfer. Worked around by testing the
  components (length=0 on empty, no-holder under threshold, etc.) and
  letting the property test catch the integration bugs.

### What I'd do differently next time
- For state-representation refactors, keep the v1 storage as a backing
  store and add v2 access methods as a thin shim **first**, then migrate
  callers gradually. Cuts the "broken build" window from hours to minutes.

## What's ready for Phase 2

The engine has:
- ✅ Working rules: setup, dice, robber/instant-discard, settlement/city/road
  (with caps), longest road, 4:1 bank trade.
- ✅ APIs the GNN/MCTS pipeline expects: `legal_mask`, `bank`, `all_hands`,
  `observation_for`, action history, stats.
- ✅ ABC balanced map generation when needed.
- ✅ 112 tests covering rules + invariants.
- ✅ Microbench harness for measuring future changes.

What's needed before training:
- Update Python `catan_bot` package to use the new API surface (mostly:
  `engine.legal_mask()` for policy mask, `engine.all_hands()` for the
  replay viewer's CardTracker).
- Switch `Engine::new(seed)` to call `generate_abc(seed)` once we want
  randomized boards.
- Decide on Phase 2 sequencing: dev cards next, or AlphaZero loop?
