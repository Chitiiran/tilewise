# Catan Engine — Design Spec (v1)

**Date:** 2026-04-27
**Status:** Approved — ready for implementation planning
**Project:** `catan_bot/` — engine component

## 1. Goal

Build a lightweight, headless, deterministic Catan game engine optimized for neural-network self-play training. The engine is a Rust crate exposed to Python via PyO3, designed to run millions of games per training run with a clean RL-friendly API.

This spec covers **only the engine**. The neural network, MCTS, training loop, and replay viewer are separate future projects that will sit on top of this engine.

## 2. Scope

### In scope (v1 — "Tier 1 / Skinny core")
- 4 players, classic 19-hex Catan board (fixed layout for v1)
- Setup phase: snake-order placement of 2 settlements + 2 roads each
- Main phase: roll → produce → build → end turn
- Builds: settlement, city, road
- Robber on a 7: move + steal one random card from a target player
- Discard half on 7 if hand size > 7 (one card at a time, player chooses)
- Win at 10 VP (settlements = 1 VP, cities = 2 VP)
- Headless operation only (no rendering)
- Deterministic: `(seed, action_sequence) → identical trajectory`
- Stats and event-log output for training observability

### Out of scope for v1 (deferred to later tiers)
- **Tier 2:** trades (bank, port), development cards, Largest Army, Longest Road, randomized board
- **Tier 3:** player↔player trading
- **Tier 4:** 5-6 player, Seafarers, Cities & Knights
- Visual rendering / replay viewer (engine produces logs; viewer is a separate program)

### Why fixed board for v1
A fixed board lets us validate the engine end-to-end without conflating "rules bugs" with "board generation bugs." Tier 2 will randomize hex resources and dice numbers (topology stays identical). Note: a fixed board may let bots memorize specific vertex placements rather than learn placement *principles* — board randomization will be one of the first Tier 2 additions for that reason.

## 3. Architecture

### 3.1 Language & Runtime
- **Engine core:** Rust crate. PyO3 bindings, built with `maturin`.
- **API wrapper:** Python (`gym`-style env, replay loading, training-side reward shaping).
- **Tests:** Rust unit + property tests; Python integration tests at the FFI boundary.

Rust from day one — no pure-Python prototype phase. Rationale: tight perf budget, exhaustive `match` on `GamePhase` catches forgotten transitions at compile time, mature toolchain (`maturin develop` for fast iteration).

### 3.2 Repository layout

```
catan_bot/
  catan_engine/                 # Rust crate
    src/
      lib.rs                    # PyO3 bindings (thin)
      engine.rs                 # Top-level orchestrator
      board.rs                  # Immutable graph: hex/vertex/edge topology
      state.rs                  # Mutable GameState
      rules.rs                  # Pure functions: legal_actions, apply
      actions.rs                # Action enum + ID encoding
      events.rs                 # GameEvent enum + EventLog ring buffer
      stats.rs                  # GameStats + fold(events) → stats
      rng.rs                    # Single seeded RNG
    tests/                      # Rust integration tests
      geometry.rs
      setup_phase.rs
      main_phase.rs
      robber.rs
      discard.rs
      legal_actions.rs
      determinism.rs
      properties.rs             # proptest-based
    benches/
      throughput.rs             # criterion benchmark
    Cargo.toml
  python/
    catan_bot/
      __init__.py               # re-exports the Rust module
      env.py                    # gym-style wrapper
      replay.py                 # load + step through saved game logs
  tests/
    test_python_api.py          # FFI contract tests
  pyproject.toml                # maturin build config
  docs/superpowers/specs/       # this file
```

### 3.3 Module responsibilities

**`board.rs` — Pure topology, immutable, built once at engine startup.**
- 19 hex nodes (resource type + dice number)
- 54 vertex nodes (settlement/city locations)
- 72 edge nodes (road locations)
- Adjacency lists: hex↔vertex, vertex↔edge, vertex↔vertex
- Reverse indexes: `dice_to_hexes[2..=12]` for fast production lookup
- No game state. Geometry only.

**`state.rs` — Everything that changes during a game.**
```rust
pub struct GameState {
    board: Arc<Board>,                // shared, never mutated
    settlements: [Option<u8>; 54],    // owner per vertex
    cities: [Option<u8>; 54],
    roads: [Option<u8>; 72],          // owner per edge
    robber_hex: u8,
    hands: [[u8; 5]; 4],              // [player][resource]
    bank: [u8; 5],                    // remaining supply
    vp: [u8; 4],
    turn: u32,
    current_player: u8,
    phase: GamePhase,
    // discard tracking, etc.
}

enum GamePhase {
    Setup1Place,    // each player places 1st settlement+road in order
    Setup2Place,    // ... 2nd in reverse order, then production
    Roll,           // current player must roll
    Main,           // current player builds / trades / ends turn
    Discard { remaining: [u8; 4] },  // post-7, players discard
    MoveRobber,     // post-7, current player moves robber
    Steal { from_options: Vec<u8> }, // current player picks steal target
    Done { winner: u8 },
}
```

`Arc<Board>` makes state-cloning cheap (the topology isn't copied), which matters if MCTS ever wants to fork-and-rollout.

**`rules.rs` — Pure functions over state.**
```rust
pub fn legal_actions(state: &GameState) -> Vec<Action>;
pub fn apply(state: &mut GameState, action: Action, rng: &mut Rng) -> Vec<GameEvent>;
pub fn is_terminal(state: &GameState) -> bool;
```
No I/O, no global state. Every rule is unit-testable by constructing a state and calling the function.

**`engine.rs` — Stateful orchestrator. The thing Python sees.**
- Holds `GameState`, `Rng`, `EventLog`, `GameStats`.
- `step(action)` validates, applies, records events, updates stats, returns `(reward, done)`.
- Handles automatic phase progression (e.g., a 7 roll → Discard → MoveRobber → Steal → Main).

**`lib.rs` — PyO3 module. Thin.** Exposes `Engine` with: `new(seed)`, `reset()`, `step(action_id)`, `legal_actions()`, `state_observation()`, `stats()`, `event_log()`.

### 3.4 Why these boundaries
- **Geometry isolated** — off-by-ones in the hex coordinate system are the most common Catan engine bug class. Tested independently with topology invariants.
- **Pure rule functions** — testable in isolation, no game-setup boilerplate per test.
- **Thin FFI surface** — small contract to maintain, easy to swap the binding tool if needed.
- **Python wrapper for env shaping** — gym wrapping, reward shaping, observation post-processing iterate weekly; don't pay Rust compile times for them.

## 4. Action space

Total: **~205 discrete action IDs**. Network output head is fixed-length.

| Action | ID range | Count |
|---|---|---|
| `BuildSettlement(vertex)` | 0–53 | 54 |
| `BuildCity(vertex)` | 54–107 | 54 |
| `BuildRoad(edge)` | 108–179 | 72 |
| `MoveRobber(hex)` | 180–198 | 19 |
| `Discard(resource)` | 199–203 | 5 |
| `EndTurn` | 204 | 1 |
| **Total** | | **205** |

### Encoding rules
- IDs are stable across versions; new actions are appended, never inserted.
- Engine internal: `enum Action { BuildSettlement(u8), ... }`. Conversion at FFI boundary.
- **Legal-action mask:** fixed-length `bool[205]`, exposed as numpy array. Network applies as `logits[~mask] = -inf` before softmax.

### Discard policy: one-at-a-time
On a 7 with hand > 7, the engine enters `Discard` phase. The current discarder picks one resource (5 actions). Engine repeats until quota met. Smaller action space than encoding all multisets, and matches how humans play.

### Robber: force-move
Standard rule: player must move robber to a *different* hex. `MoveRobber(current_hex)` is excluded from legal actions.

## 5. Observation / state encoding

The engine exposes **multiple observation views** so the consumer (CNN / GNN / transformer) picks what it needs. Cost is small (~few KB per call); flexibility is large.

### 5.1 Graph view (primary, GNN-ready)
```python
obs = {
    "hex_features":              np.ndarray,   # [19, F_hex]
    "vertex_features":           np.ndarray,   # [54, F_vert]
    "edge_features":             np.ndarray,   # [72, F_edge]
    "edge_index_hex_vertex":     np.ndarray,   # [2, E1] — fixed at reset()
    "edge_index_vertex_edge":    np.ndarray,   # [2, E2] — fixed at reset()
}
```
Heterogeneous graph: three node types, two edge types. Topology is constant across a game; sent once at `reset()` and reused.

### 5.2 2D grid view (CNN-ready)
- ~20-30 feature planes laid out on an 11×11 hex-aware grid (exact count depends on §5.5 dice encoding choice — one-hot adds ~10 planes).
- Empty cells are zero-padded. Useful for CNN baselines and ablations.

### 5.3 Scalar features (always)
Concatenated to the trunk after spatial processing:
- Current player's hand: `[wood, brick, sheep, wheat, ore]` — 5 floats
- Each opponent's hand size (counts only — hidden info): 3 floats
- Each player's VP: 4 floats
- Turn number, game phase one-hot, last dice roll: ~10 floats

### 5.4 Legal action mask (always)
- `np.ndarray` of shape `(205,)`, dtype `bool`.

### 5.5 Per-hex feature contents
- One-hot resource type (5 planes: wood/brick/sheep/wheat/ore)
- Dice number (one-hot or normalized scalar — TBD during impl)
- Robber present (1 plane)
- Has desert (1 plane)

### 5.6 Per-vertex feature contents
- Empty / settlement / city (3 mutually exclusive)
- Owner one-hot (4 planes, perspective-normalized)

### 5.7 Per-edge feature contents
- Empty / has road (2 planes)
- Owner one-hot (4 planes, perspective-normalized)

### 5.8 Perspective normalization (mandatory)
The engine emits observations from the *current player's* perspective. "Slot 0" in the observation is always the viewer; opponents fill slots 1-3 in turn order. The internal `GameState` keeps absolute player IDs 0-3; rotation happens at observation time.

This is essential — the network learns one shared policy that plays any seat, not four seat-specific policies. Standard AlphaZero pattern.

### 5.9 What about hidden info?
The observation only exposes what the current player can see:
- Their own resource hand: full counts.
- Opponents' hands: card *count* only, not contents.
- The bank: full counts (public).
- The robber: position is public.

The engine internally has perfect information; the projection filters.

## 6. Reward signal

```python
reward: float    # +1 if I just won, -1 if game ended and I didn't, 0 otherwise
done:   bool     # game terminal
```

- Sparse terminal reward, AlphaZero-style. No reward shaping in v1.
- Stored as float (not int) because the value head outputs `tanh ∈ [-1, 1]` and MCTS backups use estimated win probabilities.
- VP is a *game stat* (int), not the reward. Available via `stats.vp[player]`.

## 7. Stats / telemetry

The engine produces a `GameStats` struct, available as a snapshot at any step and finalized at game end. Modeled on the colonist.io stat screens the user provided.

### 7.1 Per-player counters
| Stat | Type |
|---|---|
| `vp_final` | u8 |
| `won` | bool |
| `settlements_built` | u32 |
| `cities_built` | u32 |
| `roads_built` | u32 |
| `resources_gained[5]` | `[u32; 5]` (wood/brick/sheep/wheat/ore) |
| `resources_gained_from_robber[5]` | `[u32; 5]` |
| `resources_lost_to_robber[5]` | `[u32; 5]` |
| `resources_lost_to_discard[5]` | `[u32; 5]` |
| `resources_spent[5]` | `[u32; 5]` |
| `cards_in_hand_max` | u32 |
| `times_robbed` | u32 |
| `robber_moves` | u32 |
| `discards_triggered` | u32 |

### 7.2 Per-game counters
| Stat | Type |
|---|---|
| `turns_played` | u32 |
| `dice_histogram[11]` | `[u32; 11]` (index 0 = roll 2, …, 10 = roll 12) |
| `seven_count` | u32 |
| `production_per_hex[19]` | `[u32; 19]` — board property, not per-player |
| `total_resources_produced` | u32 |
| `winner_player_id` | i8 (-1 if unfinished) |

`production_per_hex` is per-board, not per-player (a hex's production depends on the dice + adjacent buildings, not on which player rolled). Combined with the final board state, a helper computes `player_yield_from_hex` on demand.

### 7.3 Reserved for Tier 2 (struct fields present, unused in v1)
Dev cards drawn / played, knights played, longest road owner, largest army owner, port trades, bank trades. Reserving the layout means schema_version doesn't bump when Tier 2 lands.

### 7.4 Storage
- ~500 bytes per game.
- `schema_version: u32` field on `GameStats`. New fields bump the version; replay loaders check and degrade gracefully.

### 7.5 Update strategy
Stats are folded incrementally from the event log: each `GameEvent` mutates the running counters in O(1). `stats()` is a snapshot read.

## 8. Event log

The engine emits a typed event stream as the source of truth for stats and replay.

```rust
enum GameEvent {
    DiceRolled        { roll: u8 },
    ResourcesProduced { player: u8, hex: u8, resource: Resource, amount: u8 },
    BuildSettlement   { player: u8, vertex: u8 },
    BuildCity         { player: u8, vertex: u8 },
    BuildRoad         { player: u8, edge: u8 },
    RobberMoved       { player: u8, from_hex: u8, to_hex: u8 },
    Robbed            { from: u8, to: u8, resource: Option<Resource> },
    Discarded         { player: u8, resource: Resource },
    TurnEnded         { player: u8 },
    GameOver          { winner: u8 },
}
```

### Why an event log
- **Single source of truth** for stats. Stats are a fold over events.
- **Extensibility:** any future stat is a new fold over existing logs. We don't need to predict every analysis we'll want.
- **Replay:** `(seed, [Action]) → events` is a deterministic function.

### Cost
- ~50–100 bytes/event × ~80 events/turn × 80 turns ≈ **6 KB / game**.
- At 10k games/sec, ~60 MB/sec of events.
- Always-on (no `--stats-only` mode in v1). Persistence is opt-in: events live in a pre-allocated ring buffer; if nothing reads them, they're overwritten.

### Replay log format (versioned)
```json
{
  "schema_version": 1,
  "seed": 12345,
  "actions": [4, 27, 91, 204, ...],
  "metadata": { "engine_version": "0.1.0", "rules_tier": 1 }
}
```
~1-2 KB per game. Sufficient to reconstruct the entire trajectory because the engine is deterministic. Viewer (future project) loads a log and steps through it.

## 9. Determinism

Hard requirement: `(seed, action_sequence) → identical event log, byte-for-byte, on every machine`.

### Mechanism
- All randomness flows through one `rng.rs` `Rng` struct (seeded `SmallRng` or similar).
- No `random()`, `rand::thread_rng()`, hash-set iteration, or other implicit-order dependencies in rules code.
- `BTreeMap` over `HashMap` where iteration order matters.

### Verification
- Determinism test in `tests/determinism.rs`: hash 1000 random games for fixed seeds. Hashes are a regression target. Any change that flips a seed=42 hash is a flagged behavioral change, not silent drift.

## 10. Performance target

**10,000 games/sec, single-threaded, random vs random, on a modern laptop CPU.**

- Reference: Catanatron (pure Python) is ~1-3k. Theoretical Rust ceiling is ~50-100k. 10k is "10× Catanatron, plenty of headroom for v1."
- Per-game budget: 100 µs. Per-step budget: ~1 µs (assuming ~80 steps/game).

### Measurement
- `criterion` benchmark in `benches/throughput.rs`. Runs N=10,000 games, reports mean games/sec with confidence interval.
- CI runs the benchmark on every PR and posts the number.
- Stretch test: 4-8 worker multiprocessing should hit ~50-80k/sec linearly.

### What we are NOT optimizing for in v1
- GPU vectorization (deliberately not chosen, see options A/B/C analysis).
- Intra-game multi-threading (Catan is sequential).
- SIMD / hand-tuned inner loops.
- Memory micro-optimization (state is ~1 KB, unproblematic).

If 10k is hit easily, we chase 50k later. If not, we profile — most likely bottleneck is observation tensor construction, in which case we add a `step_no_obs()` fast path.

## 11. Testing strategy (TDD-heavy)

Strict TDD: red → green → refactor → commit. One commit per rule. ~80-100 commits expected for v1.

### 11.1 Layer 1 — Geometry tests (`tests/geometry.rs`)
Topology invariants on the immutable board:
- Vertex–hex adjacency cardinality is 1, 2, or 3.
- Edge has exactly 2 vertex endpoints.
- Handshake lemma: `Σ |adj_vertices| over hexes == Σ |adj_hexes| over vertices`.
- Specific known-good cases (vertex 0 adjacent to hexes [...]).

### 11.2 Layer 2 — Rule tests (`tests/<phase>.rs`)
Each rule in isolation. Test names describe the rule:
- `cannot_build_settlement_adjacent_to_existing`
- `rolling_8_produces_for_adjacent_settlements_and_cities`
- `rolling_7_with_8_cards_forces_discard_phase`
- `rolling_8_when_robber_on_hex_produces_nothing`
- `setup_phase_2_returns_resources_for_second_settlement`
- (~50+ rule tests expected)

### 11.3 Layer 3 — Property tests (`tests/properties.rs`, via `proptest` crate)
Generative tests over random action sequences, with automatic shrinking on failure:
- After any legal sequence, total of each resource on the board ≤ supply (19).
- No player has negative resources at any time.
- VP is non-decreasing per player in Tier 1.
- `legal_actions(state)` is non-empty unless game is over.
- Applying any legal action never panics.
- `decode(encode(action_id)) == action_id` for all 205 IDs.

### 11.4 Layer 4 — Determinism tests (`tests/determinism.rs`)
- Same seed + same actions → byte-identical event log.
- Stable hashes for 1000 fixed-seed random games — regression target.

### 11.5 Layer 5 — Python integration tests (`tests/test_python_api.py`)
FFI-boundary contract:
- `Engine.step()` types and shapes match docs.
- Observation arrays have correct shape and dtype.
- Legal mask length is exactly 205.
- Stats dict has all documented keys.

### 11.6 Performance regression (`benches/throughput.rs`)
- `criterion`-based games-per-second benchmark.
- CI fails on a 30%+ regression.

### 11.7 Top-level "smoke test"
Written first, before any rules. Fails for weeks; passing it = v1 is functionally done:
- `play_random_game_to_completion_without_panicking`

## 12. NN-readiness notes (informative)

Not engine requirements, but they justify the design choices:
- **GNN alignment:** the action space is structured. `BuildSettlement(v)` for v ∈ [0,54) maps 1-to-1 with vertex nodes. A GNN policy head can be `Linear(F_vert → 1)` per-vertex, whose 54 outputs ARE the settlement-action logits. Same for cities (per vertex) and roads (per edge). The engine just exposes per-node-type features cleanly; the network exploits the alignment.
- **Permutation invariance:** GNNs inherit board-rotation symmetries naturally. CNNs do not.
- **Heterogeneous graph:** three node types with different feature widths → use PyTorch Geometric's `HeteroData` or equivalent.

## 13. Open items deferred to implementation

These don't block the spec but will be decided during impl:
- Exact RNG choice (`SmallRng` vs `ChaCha8Rng` — speed vs quality trade-off).
- Exact dice-number encoding (one-hot vs normalized scalar).
- Whether the 11×11 grid view is layered as `(planes, H, W)` or `(H, W, planes)` (PyTorch convention is former).
- `criterion` benchmark threshold for CI failure (start with 30% regression; tighten later).

## 14. Versioning commitments

- **Action ID layout:** stable. Adding actions appends; never inserts.
- **GameStats schema:** versioned via `schema_version`. Tier 2 fields reserved.
- **Replay log format:** versioned via `schema_version`. Loader handles old versions.
- **Observation shapes:** documented contract; changes bump engine major version.

## 15. Success criteria for v1

The engine is "done" when all of the following hold:
1. `play_random_game_to_completion_without_panicking` smoke test passes for 1M games.
2. Determinism hash test passes for fixed seed set.
3. All rule unit tests pass.
4. All property tests pass at default 256 cases each.
5. Benchmark sustains ≥10,000 games/sec on a modern laptop CPU.
6. Python `gym`-style env runs random-vs-random self-play in a `for _ in range(N): env.step(env.action_space.sample(mask=env.legal_mask))` loop without errors.
7. Stats and event-log outputs match the documented schemas.
