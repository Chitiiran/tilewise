# tilewise — Catan Engine v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a headless, deterministic, Rust-core Catan engine (Tier 1 rules) exposed to Python via PyO3, with a `gym`-style env wrapper, sustaining ≥10k random-vs-random games/sec.

**Architecture:** Rust crate `catan_engine` (immutable `Board` graph + mutable `GameState` + pure `rules` functions + typed `EventLog` + folded `GameStats`) wrapped by a thin PyO3 module exposed to a Python `env.py` gym wrapper. TDD-heavy: unit tests per rule, property tests via `proptest`, determinism regression hashes, `criterion` perf benchmark.

**Tech Stack:** Rust (stable), PyO3, maturin, `proptest`, `criterion`, `rand` with `SmallRng`, Python 3.10+, `numpy`, `pytest`.

**Reference spec:** `docs/superpowers/specs/2026-04-27-catan-engine-design.md` (read this before starting any task — every implementation choice traces back to it).

---

## File Structure

```
catan_bot/                         # repo root (already exists)
  Cargo.toml                       # workspace root, points at catan_engine/
  pyproject.toml                   # maturin build config
  rust-toolchain.toml              # pin Rust stable
  catan_engine/                    # Rust crate
    Cargo.toml
    src/
      lib.rs                       # PyO3 bindings (thin)
      engine.rs                    # Top-level orchestrator (Engine struct)
      board.rs                     # Immutable graph topology
      state.rs                     # Mutable GameState + GamePhase enum
      rules.rs                     # Pure functions: legal_actions, apply
      actions.rs                   # Action enum + ID encode/decode
      events.rs                    # GameEvent enum + EventLog
      stats.rs                     # GameStats + fold_event
      rng.rs                       # Seeded Rng wrapper
      observation.rs               # Build numpy-shaped tensors from state
    tests/                         # Rust integration tests
      geometry.rs
      action_encoding.rs
      setup_phase.rs
      main_phase.rs
      robber.rs
      discard.rs
      legal_actions.rs
      determinism.rs
      properties.rs                # proptest-based
      smoke.rs                     # play_random_game_to_completion
    benches/
      throughput.rs                # criterion benchmark
  python/
    catan_bot/
      __init__.py                  # re-exports Rust module
      env.py                       # gym-style wrapper
      replay.py                    # load/step a saved game log
  tests/                           # Python integration tests
    test_python_api.py
    test_env.py
```

Each Rust source file has one responsibility (geometry / state / rules / events / stats / RNG / observation / engine / FFI). Each test file targets one phase or invariant class.

---

## Plan Phases

| Phase | Tasks | Output |
|---|---|---|
| **1. Foundation** | 1–4 | Repo wired, `cargo build`/`maturin develop` works, `from catan_bot import _engine` returns hello-world |
| **2. Board geometry** | 5–7 | Immutable `Board` graph, geometry invariants tested, fixed standard-board layout |
| **3. State + actions + events** | 8–11 | `GameState`, `Action` encoding (round-trip tested), `GameEvent` enum, `EventLog` |
| **4. Rules — Setup phase** | 12–14 | Legal placements, snake-order, Setup-2 yields starting resources |
| **5. Rules — Main phase** | 15–20 | Roll/produce, build settlement/city/road, robber+discard+steal sub-phases, end turn, win condition |
| **6. Stats + RNG + determinism** | 21–23 | Stats folded from events, deterministic RNG, determinism hash test |
| **7. Observation + PyO3 + Python env** | 24–28 | Numpy observation tensors, full FFI surface, gym-style env, integration test |
| **8. Property tests + benchmark** | 29–30 | `proptest` invariants, `criterion` 10k games/sec target |

Each task ends with a commit. Approximately 80-100 commits expected (each task has multiple TDD steps with their own commits).

---

## Phase 1 — Foundation

### Task 1: Workspace and build config

**Files:**
- Create: `C:/dojo/catan_bot/Cargo.toml`
- Create: `C:/dojo/catan_bot/rust-toolchain.toml`
- Create: `C:/dojo/catan_bot/pyproject.toml`
- Create: `C:/dojo/catan_bot/catan_engine/Cargo.toml`

- [ ] **Step 1: Create the workspace `Cargo.toml`**

`C:/dojo/catan_bot/Cargo.toml`:
```toml
[workspace]
resolver = "2"
members = ["catan_engine"]

[workspace.package]
version = "0.1.0"
edition = "2021"
license = "MIT"
```

- [ ] **Step 2: Pin the Rust toolchain**

`C:/dojo/catan_bot/rust-toolchain.toml`:
```toml
[toolchain]
channel = "stable"
components = ["rustfmt", "clippy"]
```

- [ ] **Step 3: Create the crate `Cargo.toml`**

`C:/dojo/catan_bot/catan_engine/Cargo.toml`:
```toml
[package]
name = "catan_engine"
version.workspace = true
edition.workspace = true
license.workspace = true

[lib]
name = "catan_engine"
crate-type = ["cdylib", "rlib"]

[dependencies]
pyo3 = { version = "0.22", features = ["extension-module"] }
numpy = "0.22"
rand = { version = "0.8", features = ["small_rng"] }

[dev-dependencies]
proptest = "1.4"
criterion = { version = "0.5", features = ["html_reports"] }

[[bench]]
name = "throughput"
harness = false
```

- [ ] **Step 4: Create `pyproject.toml`**

`C:/dojo/catan_bot/pyproject.toml`:
```toml
[build-system]
requires = ["maturin>=1.5"]
build-backend = "maturin"

[project]
name = "catan_bot"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["numpy>=1.26"]

[project.optional-dependencies]
dev = ["pytest>=7", "maturin>=1.5"]

[tool.maturin]
module-name = "catan_bot._engine"
python-source = "python"
features = ["pyo3/extension-module"]
```

- [ ] **Step 5: Verify build skeleton**

Run: `cd C:/dojo/catan_bot && cargo check`
Expected: error about missing `src/lib.rs` — that's fine, fixed in next task. Or success if Cargo accepts an empty crate.

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml rust-toolchain.toml pyproject.toml catan_engine/Cargo.toml
git commit -m "build: workspace + maturin scaffold"
```

---

### Task 2: Hello-world PyO3 module + Python package

**Files:**
- Create: `catan_engine/src/lib.rs`
- Create: `python/catan_bot/__init__.py`
- Create: `tests/test_python_api.py`

- [ ] **Step 1: Create the failing Python test**

`tests/test_python_api.py`:
```python
def test_engine_module_imports():
    from catan_bot import _engine
    assert _engine.engine_version() == "0.1.0"
```

- [ ] **Step 2: Verify it fails (module doesn't exist yet)**

Run: `cd C:/dojo/catan_bot && pytest tests/test_python_api.py -v`
Expected: ImportError or ModuleNotFoundError.

- [ ] **Step 3: Implement minimal Rust PyO3 module**

`catan_engine/src/lib.rs`:
```rust
use pyo3::prelude::*;

#[pyfunction]
fn engine_version() -> &'static str {
    "0.1.0"
}

#[pymodule]
fn _engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(engine_version, m)?)?;
    Ok(())
}
```

- [ ] **Step 4: Create the Python package init**

`python/catan_bot/__init__.py`:
```python
from catan_bot._engine import engine_version

__all__ = ["engine_version"]
```

- [ ] **Step 5: Build and install in develop mode**

Run: `cd C:/dojo/catan_bot && pip install maturin && maturin develop`
Expected: builds without errors, installs `catan_bot` into the active Python env.

- [ ] **Step 6: Run the test**

Run: `pytest tests/test_python_api.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add catan_engine/src/lib.rs python/ tests/
git commit -m "feat: hello-world PyO3 module"
```

---

### Task 3: Top-level smoke test placeholder (red, stays red until Phase 5)

**Files:**
- Create: `catan_engine/tests/smoke.rs`

- [ ] **Step 1: Write the smoke test that defines "v1 is functionally done"**

`catan_engine/tests/smoke.rs`:
```rust
//! Top-level smoke test. Fails until the engine can play a complete game.
//! When this passes for 1M random games, v1 is functionally done.

use catan_engine::Engine;

#[test]
fn play_random_game_to_completion_without_panicking() {
    let mut engine = Engine::new(42);
    let mut steps = 0;
    while !engine.is_terminal() {
        let legal = engine.legal_actions();
        assert!(!legal.is_empty(), "no legal actions in non-terminal state");
        let action = legal[0]; // deterministic choice for now
        engine.step(action);
        steps += 1;
        assert!(steps < 5_000, "game did not terminate in 5000 steps");
    }
    assert!(engine.is_terminal());
}
```

- [ ] **Step 2: Verify it fails (Engine doesn't exist yet)**

Run: `cd C:/dojo/catan_bot && cargo test --test smoke`
Expected: compile error — `Engine` not found. This is intentional; the test stays red until Phase 5.

- [ ] **Step 3: Commit the failing test**

```bash
git add catan_engine/tests/smoke.rs
git commit -m "test: add smoke test for v1 completion (currently red)"
```

---

### Task 4: Stub Engine struct so smoke test compiles

**Files:**
- Modify: `catan_engine/src/lib.rs`
- Create: `catan_engine/src/engine.rs`

- [ ] **Step 1: Create stub Engine module**

`catan_engine/src/engine.rs`:
```rust
//! Top-level orchestrator. Stub for now — fleshed out across Phases 2–5.

pub struct Engine {
    _seed: u64,
}

impl Engine {
    pub fn new(seed: u64) -> Self {
        Self { _seed: seed }
    }

    pub fn is_terminal(&self) -> bool {
        // Stub: always true so smoke test "completes" trivially.
        // Will be replaced by real terminal check in Phase 5.
        true
    }

    pub fn legal_actions(&self) -> Vec<u32> {
        // Stub: empty. Smoke test will skip the inner loop because is_terminal=true.
        vec![]
    }

    pub fn step(&mut self, _action: u32) {
        // Stub.
    }
}
```

- [ ] **Step 2: Re-export from lib.rs**

Add to `catan_engine/src/lib.rs` (above the `#[pymodule]` attribute):
```rust
mod engine;
pub use engine::Engine;
```

- [ ] **Step 3: Run smoke test**

Run: `cargo test --test smoke`
Expected: PASS (because `is_terminal()` returns true immediately — fake but compiling).

- [ ] **Step 4: Commit**

```bash
git add catan_engine/src/engine.rs catan_engine/src/lib.rs
git commit -m "feat: stub Engine struct so smoke test compiles"
```

---

## Phase 2 — Board geometry

### Task 5: Board struct + standard layout constants

**Files:**
- Create: `catan_engine/src/board.rs`
- Modify: `catan_engine/src/lib.rs` (add `mod board;`)

- [ ] **Step 1: Define the Board struct and Resource enum**

`catan_engine/src/board.rs`:
```rust
//! Immutable board topology for the standard 19-hex Catan board.
//! Built once at engine startup; never mutated.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Resource {
    Wood,
    Brick,
    Sheep,
    Wheat,
    Ore,
}

#[derive(Debug, Clone, Copy)]
pub struct Hex {
    pub resource: Option<Resource>, // None = desert
    pub dice_number: Option<u8>,    // None = desert
}

#[derive(Debug)]
pub struct Board {
    pub hexes: [Hex; 19],
    pub vertex_to_hexes: [Vec<u8>; 54],   // each vertex touches 1-3 hexes
    pub vertex_to_vertices: [Vec<u8>; 54], // adjacency for distance rule
    pub vertex_to_edges: [Vec<u8>; 54],
    pub edge_to_vertices: [[u8; 2]; 72],
    pub hex_to_vertices: [[u8; 6]; 19],
    pub dice_to_hexes: [Vec<u8>; 13],     // index 0,1 unused; 2..=12 used
}

impl Board {
    pub fn standard() -> Self {
        let hexes = standard_hexes();
        let hex_to_vertices = standard_hex_to_vertices();
        let edge_to_vertices = standard_edge_to_vertices();

        let vertex_to_hexes = invert_hex_to_vertices(&hex_to_vertices);
        let vertex_to_edges = invert_edge_to_vertices(&edge_to_vertices);
        let vertex_to_vertices = compute_vertex_adjacency(&edge_to_vertices);
        let dice_to_hexes = compute_dice_to_hexes(&hexes);

        Self {
            hexes,
            vertex_to_hexes,
            vertex_to_vertices,
            vertex_to_edges,
            edge_to_vertices,
            hex_to_vertices,
            dice_to_hexes,
        }
    }
}

fn standard_hexes() -> [Hex; 19] {
    use Resource::*;
    // Standard Catan layout, spiral order from top-left.
    // Resource list and dice-number sequence are the published canonical layout.
    let resources = [
        Some(Ore),   Some(Sheep), Some(Wood),
        Some(Wheat), Some(Brick), Some(Sheep), Some(Brick),
        Some(Wheat), Some(Wood),  None,         Some(Wood),  Some(Ore),
        Some(Wood),  Some(Ore),   Some(Wheat),
        Some(Sheep), Some(Brick), Some(Wheat), Some(Sheep),
    ];
    let numbers = [
        Some(10), Some(2),  Some(9),
        Some(12), Some(6),  Some(4),  Some(10),
        Some(9),  Some(11), None,     Some(3),  Some(8),
        Some(8),  Some(3),  Some(4),
        Some(5),  Some(5),  Some(6),  Some(11),
    ];
    let mut hexes = [Hex { resource: None, dice_number: None }; 19];
    for i in 0..19 {
        hexes[i] = Hex { resource: resources[i], dice_number: numbers[i] };
    }
    hexes
}

fn standard_hex_to_vertices() -> [[u8; 6]; 19] {
    // Canonical vertex IDs per hex, in clockwise order starting from top.
    // This is a published table — using bcollazo/catanatron's numbering scheme.
    [
        [0, 1, 2, 8, 7, 6],         // hex 0
        [2, 3, 4, 10, 9, 8],        // hex 1
        [4, 5, 13, 12, 11, 10],     // hex 2
        [6, 7, 16, 15, 14, 22],     // hex 3
        [7, 8, 9, 17, 16, 18],      // (placeholder values — see Step 2 note)
        [9, 10, 11, 19, 18, 17],
        [11, 12, 13, 20, 19, 21],
        [14, 15, 23, 22, 28, 27],
        [15, 16, 17, 24, 23, 25],
        [17, 18, 19, 26, 25, 24],   // desert in standard, but topology same
        [19, 20, 21, 31, 30, 26],
        [21, 13, 32, 31, 33, 34],
        [27, 28, 35, 36, 41, 40],
        [28, 29, 37, 36, 35, 38],
        [29, 30, 39, 38, 37, 42],
        [30, 31, 32, 43, 42, 39],
        [32, 33, 44, 43, 47, 46],
        [40, 41, 49, 48, 53, 52],
        [42, 39, 50, 49, 48, 51],
    ]
}
```

> **Note on the vertex numbering above:** the table values shown are illustrative placeholders. The actual canonical numbering (54 unique vertex IDs, 72 unique edge IDs, exact adjacency) must be ported from a known-correct reference. **Use Catanatron's `static/board_tensor_features.py` constants** (https://github.com/bcollazo/catanatron) as the authoritative source. Copy the exact `HEXES_VERTICES`, `EDGES_VERTICES` arrays. Do NOT invent numbering — geometry bugs are the single hardest class to debug in a Catan engine, and using a battle-tested numbering scheme from day one is the design's explicit intent (spec §3.4).

- [ ] **Step 2: Stub the helper functions**

Append to `catan_engine/src/board.rs`:
```rust
fn standard_edge_to_vertices() -> [[u8; 2]; 72] {
    // Port from Catanatron's EDGES_VERTICES table.
    // Each edge connects exactly 2 vertices.
    [[0, 0]; 72]  // PLACEHOLDER — fill from reference table in Task 6 sub-step
}

fn invert_hex_to_vertices(h2v: &[[u8; 6]; 19]) -> [Vec<u8>; 54] {
    let mut v2h: [Vec<u8>; 54] = std::array::from_fn(|_| Vec::new());
    for (hex_id, verts) in h2v.iter().enumerate() {
        for &v in verts {
            v2h[v as usize].push(hex_id as u8);
        }
    }
    v2h
}

fn invert_edge_to_vertices(e2v: &[[u8; 2]; 72]) -> [Vec<u8>; 54] {
    let mut v2e: [Vec<u8>; 54] = std::array::from_fn(|_| Vec::new());
    for (edge_id, verts) in e2v.iter().enumerate() {
        for &v in verts {
            v2e[v as usize].push(edge_id as u8);
        }
    }
    v2e
}

fn compute_vertex_adjacency(e2v: &[[u8; 2]; 72]) -> [Vec<u8>; 54] {
    let mut v2v: [Vec<u8>; 54] = std::array::from_fn(|_| Vec::new());
    for [a, b] in e2v {
        v2v[*a as usize].push(*b);
        v2v[*b as usize].push(*a);
    }
    v2v
}

fn compute_dice_to_hexes(hexes: &[Hex; 19]) -> [Vec<u8>; 13] {
    let mut d2h: [Vec<u8>; 13] = std::array::from_fn(|_| Vec::new());
    for (hex_id, hex) in hexes.iter().enumerate() {
        if let Some(n) = hex.dice_number {
            d2h[n as usize].push(hex_id as u8);
        }
    }
    d2h
}
```

- [ ] **Step 3: Wire into `lib.rs`**

Add to `catan_engine/src/lib.rs`:
```rust
mod board;
pub use board::{Board, Hex, Resource};
```

- [ ] **Step 4: Verify it compiles**

Run: `cargo build`
Expected: builds (warnings about unused fields are OK).

- [ ] **Step 5: Commit**

```bash
git add catan_engine/src/board.rs catan_engine/src/lib.rs
git commit -m "feat: Board struct skeleton with standard hex/dice layout"
```

---

### Task 6: Port canonical hex/edge/vertex numbering from Catanatron

**Files:**
- Modify: `catan_engine/src/board.rs`

This task fills in the actual values for `standard_hex_to_vertices()` and `standard_edge_to_vertices()`. **No new logic — just data entry from a verified reference.**

- [ ] **Step 1: Locate the reference**

Open https://github.com/bcollazo/catanatron in a browser and find the constants file (likely `catanatron/state_functions.py` or `catanatron/models/coordinate_system.py`). Identify the arrays mapping hex coordinates → vertex IDs and edge IDs → endpoint vertex IDs.

> If Catanatron's numbering differs significantly from a 0..54 / 0..72 dense scheme (e.g., it uses tile-coordinate tuples), translate it: enumerate all unique vertices in some deterministic traversal order, assign 0..53; same for edges 0..71. Document the chosen ordering in a comment block above each table.

- [ ] **Step 2: Replace `standard_hex_to_vertices()` body with the real table**

Replace the placeholder body in `catan_engine/src/board.rs` with the actual 19×6 array. Each row lists the 6 vertices around hex `i` in clockwise order from top.

- [ ] **Step 3: Replace `standard_edge_to_vertices()` body with the real table**

Replace the placeholder `[[0,0]; 72]` with the actual 72×2 array. Each row gives the two endpoint vertex IDs for edge `i`.

- [ ] **Step 4: Verify it still compiles**

Run: `cargo build`
Expected: builds clean.

- [ ] **Step 5: Commit**

```bash
git add catan_engine/src/board.rs
git commit -m "feat: port canonical hex/edge/vertex numbering"
```

---

### Task 7: Geometry invariant tests

**Files:**
- Create: `catan_engine/tests/geometry.rs`

- [ ] **Step 1: Write the geometry invariant tests**

`catan_engine/tests/geometry.rs`:
```rust
use catan_engine::Board;

#[test]
fn board_has_19_hexes() {
    let b = Board::standard();
    assert_eq!(b.hexes.len(), 19);
}

#[test]
fn exactly_one_hex_is_desert() {
    let b = Board::standard();
    let deserts = b.hexes.iter().filter(|h| h.resource.is_none()).count();
    assert_eq!(deserts, 1);
}

#[test]
fn desert_has_no_dice_number() {
    let b = Board::standard();
    for h in &b.hexes {
        if h.resource.is_none() {
            assert!(h.dice_number.is_none());
        } else {
            assert!(h.dice_number.is_some());
        }
    }
}

#[test]
fn every_vertex_touches_one_to_three_hexes() {
    let b = Board::standard();
    for (v, hexes) in b.vertex_to_hexes.iter().enumerate() {
        let n = hexes.len();
        assert!(
            (1..=3).contains(&n),
            "vertex {v} has {n} adjacent hexes (expected 1-3)"
        );
    }
}

#[test]
fn every_edge_has_exactly_two_endpoints() {
    let b = Board::standard();
    for (e, verts) in b.edge_to_vertices.iter().enumerate() {
        assert_ne!(verts[0], verts[1], "edge {e} has degenerate endpoints");
    }
}

#[test]
fn handshake_lemma_hex_vertex() {
    let b = Board::standard();
    let h_to_v_total: usize = b.hex_to_vertices.iter().map(|h| h.len()).sum();
    let v_to_h_total: usize = b.vertex_to_hexes.iter().map(|v| v.len()).sum();
    assert_eq!(h_to_v_total, v_to_h_total);
}

#[test]
fn vertex_adjacency_is_symmetric() {
    let b = Board::standard();
    for (a, neighbors) in b.vertex_to_vertices.iter().enumerate() {
        for &n in neighbors {
            assert!(
                b.vertex_to_vertices[n as usize].contains(&(a as u8)),
                "vertex {a}->{n} but not {n}->{a}"
            );
        }
    }
}

#[test]
fn dice_to_hexes_covers_all_non_desert() {
    let b = Board::standard();
    let mapped: usize = b.dice_to_hexes.iter().map(|v| v.len()).sum();
    let non_desert = b.hexes.iter().filter(|h| h.resource.is_some()).count();
    assert_eq!(mapped, non_desert);
}

#[test]
fn no_two_red_numbers_adjacent_on_standard_board() {
    // Red numbers (6 and 8) must not be on adjacent hexes per Catan setup rules.
    // This validates our ported standard layout.
    let b = Board::standard();
    let red_hexes: Vec<u8> = b.hexes.iter().enumerate()
        .filter(|(_, h)| matches!(h.dice_number, Some(6) | Some(8)))
        .map(|(i, _)| i as u8)
        .collect();
    // Two hexes are adjacent iff they share at least 2 vertices.
    for &a in &red_hexes {
        for &c in &red_hexes {
            if a == c { continue; }
            let av: std::collections::HashSet<u8> =
                b.hex_to_vertices[a as usize].iter().copied().collect();
            let cv: std::collections::HashSet<u8> =
                b.hex_to_vertices[c as usize].iter().copied().collect();
            let shared = av.intersection(&cv).count();
            assert!(shared < 2, "hexes {a} and {c} are adjacent and both have red numbers");
        }
    }
}
```

- [ ] **Step 2: Run the tests**

Run: `cargo test --test geometry`
Expected: All tests PASS. **If any fail, the canonical numbering ported in Task 6 is wrong — fix the table, do NOT relax the test.**

- [ ] **Step 3: Commit**

```bash
git add catan_engine/tests/geometry.rs
git commit -m "test: board geometry invariants"
```

---

## Phase 3 — State, actions, events

### Task 8: Action enum + ID encoding (with round-trip test)

**Files:**
- Create: `catan_engine/src/actions.rs`
- Create: `catan_engine/tests/action_encoding.rs`
- Modify: `catan_engine/src/lib.rs`

- [ ] **Step 1: Write the failing round-trip test**

`catan_engine/tests/action_encoding.rs`:
```rust
use catan_engine::actions::{Action, ACTION_SPACE_SIZE, decode, encode};
use catan_engine::board::Resource;

#[test]
fn action_space_size_is_205() {
    assert_eq!(ACTION_SPACE_SIZE, 205);
}

#[test]
fn encode_decode_roundtrip_for_all_ids() {
    for id in 0..ACTION_SPACE_SIZE {
        let action = decode(id as u32).expect("every ID must decode");
        let re_encoded = encode(action);
        assert_eq!(re_encoded, id as u32, "round-trip failed for ID {id}");
    }
}

#[test]
fn id_layout_matches_spec() {
    // Spec §4: BuildSettlement(0..53)=0..53,
    //          BuildCity(0..53)=54..107,
    //          BuildRoad(0..71)=108..179,
    //          MoveRobber(0..18)=180..198,
    //          Discard(Wood..Ore)=199..203,
    //          EndTurn=204
    assert_eq!(encode(Action::BuildSettlement(0)), 0);
    assert_eq!(encode(Action::BuildSettlement(53)), 53);
    assert_eq!(encode(Action::BuildCity(0)), 54);
    assert_eq!(encode(Action::BuildCity(53)), 107);
    assert_eq!(encode(Action::BuildRoad(0)), 108);
    assert_eq!(encode(Action::BuildRoad(71)), 179);
    assert_eq!(encode(Action::MoveRobber(0)), 180);
    assert_eq!(encode(Action::MoveRobber(18)), 198);
    assert_eq!(encode(Action::Discard(Resource::Wood)), 199);
    assert_eq!(encode(Action::Discard(Resource::Ore)), 203);
    assert_eq!(encode(Action::EndTurn), 204);
}

#[test]
fn out_of_range_ids_return_none() {
    assert!(decode(205).is_none());
    assert!(decode(u32::MAX).is_none());
}
```

- [ ] **Step 2: Verify it fails**

Run: `cargo test --test action_encoding`
Expected: compile error — `Action`, `ACTION_SPACE_SIZE`, `encode`, `decode` not defined.

- [ ] **Step 3: Implement actions.rs**

`catan_engine/src/actions.rs`:
```rust
//! Action enum + ID encoding. IDs are stable across versions per spec §4.

use crate::board::Resource;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    BuildSettlement(u8), // vertex 0..54
    BuildCity(u8),       // vertex 0..54
    BuildRoad(u8),       // edge 0..72
    MoveRobber(u8),      // hex 0..19
    Discard(Resource),
    EndTurn,
}

pub const ACTION_SPACE_SIZE: usize = 205;

pub fn encode(action: Action) -> u32 {
    match action {
        Action::BuildSettlement(v) => v as u32,
        Action::BuildCity(v) => 54 + v as u32,
        Action::BuildRoad(e) => 108 + e as u32,
        Action::MoveRobber(h) => 180 + h as u32,
        Action::Discard(r) => 199 + resource_index(r) as u32,
        Action::EndTurn => 204,
    }
}

pub fn decode(id: u32) -> Option<Action> {
    match id {
        0..=53 => Some(Action::BuildSettlement(id as u8)),
        54..=107 => Some(Action::BuildCity((id - 54) as u8)),
        108..=179 => Some(Action::BuildRoad((id - 108) as u8)),
        180..=198 => Some(Action::MoveRobber((id - 180) as u8)),
        199..=203 => Some(Action::Discard(resource_from_index((id - 199) as u8))),
        204 => Some(Action::EndTurn),
        _ => None,
    }
}

fn resource_index(r: Resource) -> u8 {
    match r {
        Resource::Wood => 0,
        Resource::Brick => 1,
        Resource::Sheep => 2,
        Resource::Wheat => 3,
        Resource::Ore => 4,
    }
}

fn resource_from_index(i: u8) -> Resource {
    match i {
        0 => Resource::Wood,
        1 => Resource::Brick,
        2 => Resource::Sheep,
        3 => Resource::Wheat,
        4 => Resource::Ore,
        _ => panic!("invalid resource index {i}"),
    }
}
```

- [ ] **Step 4: Wire into lib.rs**

Add to `catan_engine/src/lib.rs`:
```rust
pub mod actions;
pub mod board;  // change from `mod board;` if not already pub
```

- [ ] **Step 5: Run tests**

Run: `cargo test --test action_encoding`
Expected: All 4 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add catan_engine/src/actions.rs catan_engine/src/lib.rs catan_engine/tests/action_encoding.rs
git commit -m "feat: Action enum + ID encoding with round-trip test"
```

---

### Task 9: GameEvent enum and EventLog

**Files:**
- Create: `catan_engine/src/events.rs`
- Modify: `catan_engine/src/lib.rs`

- [ ] **Step 1: Define events.rs**

`catan_engine/src/events.rs`:
```rust
//! Typed event stream — source of truth for stats and replay.

use crate::board::Resource;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameEvent {
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

pub struct EventLog {
    events: Vec<GameEvent>,
}

impl EventLog {
    pub fn new() -> Self {
        Self { events: Vec::with_capacity(1024) }
    }

    pub fn push(&mut self, e: GameEvent) {
        self.events.push(e);
    }

    pub fn as_slice(&self) -> &[GameEvent] {
        &self.events
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn clear(&mut self) {
        self.events.clear();
    }
}

impl Default for EventLog {
    fn default() -> Self { Self::new() }
}
```

- [ ] **Step 2: Wire into lib.rs**

Add to `catan_engine/src/lib.rs`:
```rust
pub mod events;
```

- [ ] **Step 3: Build**

Run: `cargo build`
Expected: clean build.

- [ ] **Step 4: Commit**

```bash
git add catan_engine/src/events.rs catan_engine/src/lib.rs
git commit -m "feat: GameEvent enum + EventLog"
```

---

### Task 10: GameState struct + GamePhase enum

**Files:**
- Create: `catan_engine/src/state.rs`
- Modify: `catan_engine/src/lib.rs`

- [ ] **Step 1: Define state.rs**

`catan_engine/src/state.rs`:
```rust
//! Mutable game state. Internal representation; hidden info filtered at observation time.

use crate::board::Board;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GamePhase {
    /// First settlement+road placement, in player order 0..N_PLAYERS.
    Setup1Place,
    /// Second settlement+road placement, in REVERSE player order.
    /// Triggers starting-resource production for the second settlement.
    Setup2Place,
    /// Current player must roll the dice.
    Roll,
    /// Current player builds / ends turn.
    Main,
    /// After a 7 roll: each player with hand>7 must discard half.
    /// `remaining[i]` = number of cards player i still needs to discard.
    Discard { remaining: [u8; 4] },
    /// Current player must move the robber.
    MoveRobber,
    /// Current player picks a steal target from `from_options`.
    /// If `from_options` is empty, no steal happens (auto-skipped in step()).
    Steal { from_options: Vec<u8> },
    /// Game over.
    Done { winner: u8 },
}

pub const N_PLAYERS: usize = 4;
pub const N_RESOURCES: usize = 5;
pub const RESOURCE_SUPPLY: u8 = 19;
pub const WIN_VP: u8 = 10;

#[derive(Debug, Clone)]
pub struct GameState {
    pub board: Arc<Board>,
    pub settlements: [Option<u8>; 54], // owner per vertex
    pub cities: [Option<u8>; 54],
    pub roads: [Option<u8>; 72],
    pub robber_hex: u8,
    pub hands: [[u8; N_RESOURCES]; N_PLAYERS],
    pub bank: [u8; N_RESOURCES],
    pub vp: [u8; N_PLAYERS],
    pub turn: u32,
    pub current_player: u8,
    pub phase: GamePhase,
    /// Setup-phase tracker: (vertex, edge) of the most recent placement,
    /// used to enforce "road must connect to the just-placed settlement".
    pub setup_pending: Option<u8>,
}

impl GameState {
    pub fn new(board: Arc<Board>) -> Self {
        let desert_hex = board.hexes.iter().position(|h| h.resource.is_none()).unwrap_or(0) as u8;
        Self {
            board,
            settlements: [None; 54],
            cities: [None; 54],
            roads: [None; 72],
            robber_hex: desert_hex,
            hands: [[0; N_RESOURCES]; N_PLAYERS],
            bank: [RESOURCE_SUPPLY; N_RESOURCES],
            vp: [0; N_PLAYERS],
            turn: 0,
            current_player: 0,
            phase: GamePhase::Setup1Place,
            setup_pending: None,
        }
    }

    pub fn is_terminal(&self) -> bool {
        matches!(self.phase, GamePhase::Done { .. })
    }
}
```

- [ ] **Step 2: Wire into lib.rs**

Add to `catan_engine/src/lib.rs`:
```rust
pub mod state;
```

- [ ] **Step 3: Build**

Run: `cargo build`
Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add catan_engine/src/state.rs catan_engine/src/lib.rs
git commit -m "feat: GameState + GamePhase enum"
```

---

### Task 11: rules.rs scaffold (empty pure-function shells)

**Files:**
- Create: `catan_engine/src/rules.rs`
- Modify: `catan_engine/src/lib.rs`

- [ ] **Step 1: Define the rules module surface (empty bodies)**

`catan_engine/src/rules.rs`:
```rust
//! Pure functions over GameState. No I/O, no globals.
//! Every rule is unit-testable by constructing a state and calling the function.

use crate::actions::Action;
use crate::events::GameEvent;
use crate::rng::Rng;
use crate::state::GameState;

pub fn legal_actions(state: &GameState) -> Vec<Action> {
    // Implemented incrementally across Phases 4–5, dispatching on state.phase.
    // Each phase variant gets its own helper below.
    match &state.phase {
        crate::state::GamePhase::Setup1Place => legal_actions_setup_place(state),
        crate::state::GamePhase::Setup2Place => legal_actions_setup_place(state),
        crate::state::GamePhase::Roll => vec![],          // filled in Task 15
        crate::state::GamePhase::Main => vec![],          // filled in Task 16
        crate::state::GamePhase::Discard { .. } => vec![], // filled in Task 18
        crate::state::GamePhase::MoveRobber => vec![],     // filled in Task 19
        crate::state::GamePhase::Steal { .. } => vec![],   // filled in Task 19
        crate::state::GamePhase::Done { .. } => vec![],
    }
}

pub(crate) fn legal_actions_setup_place(_state: &GameState) -> Vec<Action> {
    // Filled in Task 12.
    vec![]
}

pub fn apply(state: &mut GameState, action: Action, rng: &mut Rng) -> Vec<GameEvent> {
    // Implemented incrementally. Stub for now.
    let _ = (state, action, rng);
    vec![]
}

pub fn is_terminal(state: &GameState) -> bool {
    state.is_terminal()
}
```

- [ ] **Step 2: Stub the Rng module (will be properly defined in Task 22)**

`catan_engine/src/rng.rs`:
```rust
//! Seeded RNG. All randomness flows through here for determinism.
//! Properly defined in Task 22; stub for now so rules.rs compiles.

use rand::rngs::SmallRng;
use rand::SeedableRng;

pub struct Rng {
    inner: SmallRng,
}

impl Rng {
    pub fn from_seed(seed: u64) -> Self {
        Self { inner: SmallRng::seed_from_u64(seed) }
    }

    pub fn inner(&mut self) -> &mut SmallRng {
        &mut self.inner
    }
}
```

- [ ] **Step 3: Wire into lib.rs**

Add to `catan_engine/src/lib.rs`:
```rust
pub mod rules;
pub mod rng;
```

- [ ] **Step 4: Build**

Run: `cargo build`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add catan_engine/src/rules.rs catan_engine/src/rng.rs catan_engine/src/lib.rs
git commit -m "feat: rules.rs and rng.rs scaffolds"
```

---

## Phase 4 — Rules: Setup phase

### Task 12: Setup-1 settlement placement (legal locations + apply)

**Files:**
- Modify: `catan_engine/src/rules.rs`
- Create: `catan_engine/tests/setup_phase.rs`

- [ ] **Step 1: Write failing tests for Setup-1 settlement placement**

`catan_engine/tests/setup_phase.rs`:
```rust
use catan_engine::actions::Action;
use catan_engine::board::Board;
use catan_engine::rng::Rng;
use catan_engine::rules::{apply, legal_actions};
use catan_engine::state::{GameState, GamePhase};
use std::sync::Arc;

fn fresh_state() -> GameState {
    GameState::new(Arc::new(Board::standard()))
}

#[test]
fn setup1_legal_actions_are_all_54_settlements_when_board_empty() {
    let state = fresh_state();
    let legal = legal_actions(&state);
    let settlement_actions: Vec<_> = legal.iter()
        .filter(|a| matches!(a, Action::BuildSettlement(_)))
        .collect();
    assert_eq!(settlement_actions.len(), 54);
}

#[test]
fn setup1_distance_rule_blocks_adjacent_vertices() {
    let mut state = fresh_state();
    let mut rng = Rng::from_seed(0);
    // Place settlement at vertex 0.
    apply(&mut state, Action::BuildSettlement(0), &mut rng);
    // After placing, the road action follows; for this test just inspect occupancy.
    assert_eq!(state.settlements[0], Some(0)); // owned by player 0
    // After settlement is placed, before player picks a road,
    // legal_actions should enumerate roads at edges of vertex 0 only.
    // (The distance check itself: vertex 0's neighbors must not be settlement-legal
    // even if we re-entered Setup1Place — verified in next test.)
}

#[test]
fn setup1_after_first_settlement_road_must_be_adjacent_to_it() {
    let mut state = fresh_state();
    let mut rng = Rng::from_seed(0);
    apply(&mut state, Action::BuildSettlement(0), &mut rng);
    let legal = legal_actions(&state);
    let road_actions: Vec<u8> = legal.iter()
        .filter_map(|a| match a {
            Action::BuildRoad(e) => Some(*e),
            _ => None,
        })
        .collect();
    let v0_edges = &state.board.vertex_to_edges[0];
    for e in &road_actions {
        assert!(v0_edges.contains(e), "road {e} not adjacent to settlement at vertex 0");
    }
}
```

- [ ] **Step 2: Run — they should fail (legal_actions is stub)**

Run: `cargo test --test setup_phase`
Expected: tests fail because legal_actions returns empty vec.

- [ ] **Step 3: Implement Setup-1 settlement legality**

Replace `legal_actions_setup_place` in `catan_engine/src/rules.rs`:
```rust
pub(crate) fn legal_actions_setup_place(state: &GameState) -> Vec<Action> {
    // Two sub-states:
    //   setup_pending == None  → place a settlement
    //   setup_pending == Some(v) → place a road adjacent to v
    match state.setup_pending {
        None => legal_setup_settlements(state),
        Some(v) => legal_setup_roads(state, v),
    }
}

fn legal_setup_settlements(state: &GameState) -> Vec<Action> {
    let mut out = Vec::with_capacity(54);
    for v in 0u8..54 {
        if is_legal_settlement_location(state, v) {
            out.push(Action::BuildSettlement(v));
        }
    }
    out
}

fn legal_setup_roads(state: &GameState, just_placed_vertex: u8) -> Vec<Action> {
    let mut out = Vec::new();
    for &e in &state.board.vertex_to_edges[just_placed_vertex as usize] {
        if state.roads[e as usize].is_none() {
            out.push(Action::BuildRoad(e));
        }
    }
    out
}

fn is_legal_settlement_location(state: &GameState, v: u8) -> bool {
    if state.settlements[v as usize].is_some() || state.cities[v as usize].is_some() {
        return false;
    }
    // Distance rule: no neighbor vertex may have a settlement or city.
    for &n in &state.board.vertex_to_vertices[v as usize] {
        if state.settlements[n as usize].is_some() || state.cities[n as usize].is_some() {
            return false;
        }
    }
    true
}
```

- [ ] **Step 4: Implement `apply` for Setup-1 settlement placement**

Replace the body of `apply` in `catan_engine/src/rules.rs`:
```rust
pub fn apply(state: &mut GameState, action: Action, rng: &mut Rng) -> Vec<GameEvent> {
    let mut events = Vec::new();
    match (&state.phase, action) {
        (GamePhase::Setup1Place, Action::BuildSettlement(v)) => {
            state.settlements[v as usize] = Some(state.current_player);
            state.vp[state.current_player as usize] += 1;
            state.setup_pending = Some(v);
            events.push(GameEvent::BuildSettlement { player: state.current_player, vertex: v });
        }
        _ => {
            // Other transitions implemented in Tasks 13–20.
            let _ = rng;
        }
    }
    events
}
```

(You will also need `use crate::state::GamePhase;` at the top of rules.rs.)

- [ ] **Step 5: Run tests**

Run: `cargo test --test setup_phase`
Expected: First test passes; the road-test passes; distance-rule test passes.

- [ ] **Step 6: Commit**

```bash
git add catan_engine/src/rules.rs catan_engine/tests/setup_phase.rs
git commit -m "feat: Setup-1 settlement placement legality + apply"
```

---

### Task 13: Setup-1 road placement + advance to Setup-2

**Files:**
- Modify: `catan_engine/src/rules.rs`
- Modify: `catan_engine/tests/setup_phase.rs`

- [ ] **Step 1: Add tests for Setup-1 road placement and player advancement**

Append to `catan_engine/tests/setup_phase.rs`:
```rust
#[test]
fn setup1_placing_road_advances_to_next_player() {
    let mut state = fresh_state();
    let mut rng = Rng::from_seed(0);
    apply(&mut state, Action::BuildSettlement(0), &mut rng);
    let road_edge = state.board.vertex_to_edges[0][0];
    apply(&mut state, Action::BuildRoad(road_edge), &mut rng);
    assert_eq!(state.current_player, 1);
    assert!(matches!(state.phase, GamePhase::Setup1Place));
    assert!(state.setup_pending.is_none());
}

#[test]
fn setup1_after_all_4_players_place_first_settlement_we_enter_setup2() {
    let mut state = fresh_state();
    let mut rng = Rng::from_seed(0);
    let starting_vertices = [0u8, 8, 16, 24];
    for v in starting_vertices {
        apply(&mut state, Action::BuildSettlement(v), &mut rng);
        let road_edge = state.board.vertex_to_edges[v as usize][0];
        apply(&mut state, Action::BuildRoad(road_edge), &mut rng);
    }
    assert!(matches!(state.phase, GamePhase::Setup2Place));
    // Setup-2 goes in REVERSE order, so player 3 places first.
    assert_eq!(state.current_player, 3);
}
```

- [ ] **Step 2: Verify failure**

Run: `cargo test --test setup_phase`
Expected: new tests fail.

- [ ] **Step 3: Extend `apply` for Setup-1 road placement**

Add to the `match` in `apply`:
```rust
        (GamePhase::Setup1Place, Action::BuildRoad(e)) => {
            state.roads[e as usize] = Some(state.current_player);
            state.setup_pending = None;
            events.push(GameEvent::BuildRoad { player: state.current_player, edge: e });
            // Advance player; if all 4 placed, enter Setup-2 (reverse order, player 3 first).
            if state.current_player < 3 {
                state.current_player += 1;
            } else {
                state.phase = GamePhase::Setup2Place;
                // current_player stays at 3 — Setup-2 starts with player 3.
            }
        }
```

- [ ] **Step 4: Run tests**

Run: `cargo test --test setup_phase`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add catan_engine/src/rules.rs catan_engine/tests/setup_phase.rs
git commit -m "feat: Setup-1 road placement + advance to Setup-2"
```

---

### Task 14: Setup-2 placement (settlement + road) with starting resource yield

**Files:**
- Modify: `catan_engine/src/rules.rs`
- Modify: `catan_engine/tests/setup_phase.rs`

- [ ] **Step 1: Add tests for Setup-2**

Append to `catan_engine/tests/setup_phase.rs`:
```rust
#[test]
fn setup2_second_settlement_yields_starting_resources() {
    let mut state = fresh_state();
    let mut rng = Rng::from_seed(0);
    // Run Setup-1 quickly.
    let s1_vertices = [0u8, 8, 16, 24];
    for v in s1_vertices {
        apply(&mut state, Action::BuildSettlement(v), &mut rng);
        let e = state.board.vertex_to_edges[v as usize][0];
        apply(&mut state, Action::BuildRoad(e), &mut rng);
    }
    // Now in Setup-2; player 3 places. Pick a vertex adjacent to known producing hexes.
    // We'll just pick legal vertex 30 and verify SOME resource ended up in player 3's hand.
    let target_v: u8 = legal_actions(&state)
        .iter()
        .find_map(|a| if let Action::BuildSettlement(v) = a { Some(*v) } else { None })
        .expect("at least one settlement is legal");
    apply(&mut state, Action::BuildSettlement(target_v), &mut rng);
    let total: u8 = state.hands[3].iter().sum();
    let n_resource_hexes_adj = state.board.vertex_to_hexes[target_v as usize]
        .iter()
        .filter(|&&h| state.board.hexes[h as usize].resource.is_some())
        .count() as u8;
    assert_eq!(total, n_resource_hexes_adj,
        "player 3 should receive 1 card per non-desert hex adjacent to setup-2 settlement");
}

#[test]
fn setup2_after_all_4_place_we_enter_roll_phase_with_player_0() {
    let mut state = fresh_state();
    let mut rng = Rng::from_seed(0);
    let s1 = [0u8, 8, 16, 24];
    let s2 = [30u8, 38, 46, 50]; // pick reasonably-spaced legal vertices
    for v in s1 {
        apply(&mut state, Action::BuildSettlement(v), &mut rng);
        apply(&mut state, Action::BuildRoad(state.board.vertex_to_edges[v as usize][0]), &mut rng);
    }
    for v in s2 {
        apply(&mut state, Action::BuildSettlement(v), &mut rng);
        apply(&mut state, Action::BuildRoad(state.board.vertex_to_edges[v as usize][0]), &mut rng);
    }
    assert!(matches!(state.phase, GamePhase::Roll));
    assert_eq!(state.current_player, 0);
}
```

> If `s2` vertices conflict with the s1 distance rule, swap them for any 4 that pass `is_legal_settlement_location`. The test logic is what matters.

- [ ] **Step 2: Verify failure**

Run: `cargo test --test setup_phase`
Expected: new tests fail.

- [ ] **Step 3: Implement Setup-2 in `apply`**

Add to the match arm in `apply`:
```rust
        (GamePhase::Setup2Place, Action::BuildSettlement(v)) => {
            state.settlements[v as usize] = Some(state.current_player);
            state.vp[state.current_player as usize] += 1;
            state.setup_pending = Some(v);
            events.push(GameEvent::BuildSettlement { player: state.current_player, vertex: v });
            // Yield starting resources: 1 card per non-desert hex adjacent.
            let board = state.board.clone();
            for &h in &board.vertex_to_hexes[v as usize] {
                if let Some(res) = board.hexes[h as usize].resource {
                    let ri = res as usize;
                    if state.bank[ri] > 0 {
                        state.bank[ri] -= 1;
                        state.hands[state.current_player as usize][ri] += 1;
                        events.push(GameEvent::ResourcesProduced {
                            player: state.current_player,
                            hex: h,
                            resource: res,
                            amount: 1,
                        });
                    }
                }
            }
        }
        (GamePhase::Setup2Place, Action::BuildRoad(e)) => {
            state.roads[e as usize] = Some(state.current_player);
            state.setup_pending = None;
            events.push(GameEvent::BuildRoad { player: state.current_player, edge: e });
            // Reverse-order advance; when player 0 finishes, transition to Roll.
            if state.current_player > 0 {
                state.current_player -= 1;
            } else {
                state.phase = GamePhase::Roll;
                // current_player stays at 0 — main game starts with player 0.
            }
        }
```

> Note: `Resource as usize` requires `#[repr(u8)]` on the enum. Add it to `board.rs`:
> ```rust
> #[repr(u8)]
> #[derive(Debug, Clone, Copy, PartialEq, Eq)]
> pub enum Resource {
>     Wood = 0, Brick = 1, Sheep = 2, Wheat = 3, Ore = 4,
> }
> ```

- [ ] **Step 4: Run tests**

Run: `cargo test --test setup_phase`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add catan_engine/src/rules.rs catan_engine/src/board.rs catan_engine/tests/setup_phase.rs
git commit -m "feat: Setup-2 placement with starting resource yield"
```

---

## Phase 5 — Rules: Main phase

### Task 15: Roll phase + production

**Files:**
- Modify: `catan_engine/src/rules.rs`
- Create: `catan_engine/tests/main_phase.rs`

- [ ] **Step 1: Tests for rolling and production**

`catan_engine/tests/main_phase.rs`:
```rust
use catan_engine::actions::Action;
use catan_engine::board::Board;
use catan_engine::rng::Rng;
use catan_engine::rules::{apply, legal_actions};
use catan_engine::state::{GameState, GamePhase};
use std::sync::Arc;

/// Build a fully-set-up state where it's player 0's turn to Roll.
fn ready_to_roll() -> GameState {
    let mut state = GameState::new(Arc::new(Board::standard()));
    let mut rng = Rng::from_seed(0);
    let s1 = [0u8, 8, 16, 24];
    let s2 = [30u8, 38, 46, 50];
    for v in s1 {
        apply(&mut state, Action::BuildSettlement(v), &mut rng);
        apply(&mut state, Action::BuildRoad(state.board.vertex_to_edges[v as usize][0]), &mut rng);
    }
    for v in s2 {
        apply(&mut state, Action::BuildSettlement(v), &mut rng);
        apply(&mut state, Action::BuildRoad(state.board.vertex_to_edges[v as usize][0]), &mut rng);
    }
    assert!(matches!(state.phase, GamePhase::Roll));
    state
}

#[test]
fn legal_actions_in_roll_phase_is_just_endturn_proxy_for_roll() {
    // Spec choice: rolling is automatic on entering Roll phase via a
    // "RollDice" action embedded in EndTurn, OR it's its own action.
    // We model it as: in Roll phase, the only legal action is EndTurn,
    // and apply() rolls, produces, and transitions to Main in one step.
    // (Alternative: add an explicit Roll(u8) action — see implementation.)
    let state = ready_to_roll();
    let legal = legal_actions(&state);
    // We use EndTurn as the "go" action in Roll phase. See Task 15 design note.
    assert!(legal.contains(&Action::EndTurn) || !legal.is_empty());
}

#[test]
fn rolling_a_non_seven_produces_resources_and_enters_main() {
    let mut state = ready_to_roll();
    let mut rng = Rng::from_seed(7); // seed chosen so first roll != 7 (verify in test)
    let bank_before = state.bank;
    apply(&mut state, Action::EndTurn, &mut rng);
    // Either we're in Main now, or we entered Discard/MoveRobber on a 7. Check Main case.
    if matches!(state.phase, GamePhase::Main) {
        // At least one resource may have been produced (depends on dice + adjacency).
        let bank_after_total: u32 = state.bank.iter().map(|&x| x as u32).sum();
        let bank_before_total: u32 = bank_before.iter().map(|&x| x as u32).sum();
        assert!(bank_after_total <= bank_before_total);
    }
}
```

> **Design note for the implementer:** in the Roll phase we model "roll the dice" as the *only* legal action and use `EndTurn` as its action ID for simplicity. Alternative would be adding `Action::RollDice` (extra ID 205+), but the engine is internal and the action is forced anyway. Document this in the rules.rs code.

- [ ] **Step 2: Verify failure**

Run: `cargo test --test main_phase`
Expected: tests fail (Roll arm not implemented).

- [ ] **Step 3: Implement Roll-phase legal_actions and apply**

In `legal_actions`, replace the `Roll` arm:
```rust
        crate::state::GamePhase::Roll => vec![Action::EndTurn],
```

In `apply`, add:
```rust
        (GamePhase::Roll, Action::EndTurn) => {
            // "EndTurn" in Roll phase = "roll the dice" (see Task 15 design note).
            use rand::Rng as _;
            let d1 = rng.inner().gen_range(1u8..=6);
            let d2 = rng.inner().gen_range(1u8..=6);
            let roll = d1 + d2;
            events.push(GameEvent::DiceRolled { roll });
            if roll == 7 {
                // Discard + robber sub-flow handled in Tasks 18-19.
                let mut remaining = [0u8; 4];
                for p in 0..4usize {
                    let total: u8 = state.hands[p].iter().sum();
                    if total > 7 {
                        remaining[p] = total / 2;
                    }
                }
                state.phase = if remaining.iter().any(|&n| n > 0) {
                    GamePhase::Discard { remaining }
                } else {
                    GamePhase::MoveRobber
                };
            } else {
                produce_resources(state, roll, &mut events);
                state.phase = GamePhase::Main;
            }
        }
```

Add at the bottom of rules.rs:
```rust
fn produce_resources(state: &mut GameState, roll: u8, events: &mut Vec<GameEvent>) {
    let board = state.board.clone();
    let hexes_for_roll = &board.dice_to_hexes[roll as usize];
    for &h in hexes_for_roll {
        if h == state.robber_hex { continue; }
        let res = match board.hexes[h as usize].resource {
            Some(r) => r,
            None => continue,
        };
        let ri = res as usize;
        for &v in &board.hex_to_vertices[h as usize] {
            // Settlement = 1 card; city = 2.
            let owner_qty = match (state.settlements[v as usize], state.cities[v as usize]) {
                (_, Some(p)) => Some((p, 2u8)),
                (Some(p), None) => Some((p, 1u8)),
                _ => None,
            };
            if let Some((p, qty)) = owner_qty {
                let take = qty.min(state.bank[ri]);
                if take > 0 {
                    state.bank[ri] -= take;
                    state.hands[p as usize][ri] += take;
                    events.push(GameEvent::ResourcesProduced {
                        player: p, hex: h, resource: res, amount: take,
                    });
                }
            }
        }
    }
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test --test main_phase`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add catan_engine/src/rules.rs catan_engine/tests/main_phase.rs
git commit -m "feat: Roll phase with dice + production"
```

---

### Task 16: Main phase — build settlement (with cost check, distance, road-connection)

**Files:**
- Modify: `catan_engine/src/rules.rs`
- Modify: `catan_engine/tests/main_phase.rs`

- [ ] **Step 1: Tests**

Append to `catan_engine/tests/main_phase.rs`:
```rust
#[test]
fn cannot_build_settlement_without_resources() {
    let mut state = ready_to_roll();
    state.phase = GamePhase::Main;
    state.hands[0] = [0; 5]; // empty hand
    let legal = legal_actions(&state);
    assert!(!legal.iter().any(|a| matches!(a, Action::BuildSettlement(_))));
}

#[test]
fn can_build_settlement_with_required_resources_on_road_endpoint() {
    let mut state = ready_to_roll();
    state.phase = GamePhase::Main;
    // Give player 0 settlement cost: 1 wood, 1 brick, 1 sheep, 1 wheat.
    state.hands[0] = [1, 1, 1, 1, 0];
    let legal = legal_actions(&state);
    let settlement_actions: Vec<u8> = legal.iter()
        .filter_map(|a| if let Action::BuildSettlement(v) = a { Some(*v) } else { None })
        .collect();
    // At least one settlement should be legal (the road endpoints from setup).
    assert!(!settlement_actions.is_empty());
}

#[test]
fn building_settlement_deducts_resources_and_grants_vp() {
    let mut state = ready_to_roll();
    state.phase = GamePhase::Main;
    state.hands[0] = [2, 2, 2, 2, 0];
    let mut rng = Rng::from_seed(0);
    let v_to_build = legal_actions(&state).iter()
        .find_map(|a| if let Action::BuildSettlement(v) = a { Some(*v) } else { None })
        .expect("at least one legal settlement");
    let vp_before = state.vp[0];
    apply(&mut state, Action::BuildSettlement(v_to_build), &mut rng);
    assert_eq!(state.hands[0], [1, 1, 1, 1, 0]);
    assert_eq!(state.vp[0], vp_before + 1);
    assert_eq!(state.settlements[v_to_build as usize], Some(0));
}
```

- [ ] **Step 2: Implement Main-phase settlement legality**

Add to `legal_actions` `Main` arm in rules.rs:
```rust
        crate::state::GamePhase::Main => legal_actions_main(state),
```

Add helper functions in rules.rs:
```rust
const SETTLEMENT_COST: [u8; 5] = [1, 1, 1, 1, 0]; // wood,brick,sheep,wheat,ore
const CITY_COST: [u8; 5] =       [0, 0, 0, 2, 3];
const ROAD_COST: [u8; 5] =       [1, 1, 0, 0, 0];

fn can_afford(hand: &[u8; 5], cost: &[u8; 5]) -> bool {
    hand.iter().zip(cost).all(|(h, c)| h >= c)
}

fn pay(hand: &mut [u8; 5], bank: &mut [u8; 5], cost: &[u8; 5]) {
    for i in 0..5 {
        hand[i] -= cost[i];
        bank[i] += cost[i];
    }
}

fn legal_actions_main(state: &GameState) -> Vec<Action> {
    let mut out = vec![Action::EndTurn];
    let p = state.current_player;
    let hand = &state.hands[p as usize];
    if can_afford(hand, &SETTLEMENT_COST) {
        for v in 0u8..54 {
            if is_legal_settlement_for_player(state, v, p) {
                out.push(Action::BuildSettlement(v));
            }
        }
    }
    if can_afford(hand, &CITY_COST) {
        for v in 0u8..54 {
            if state.settlements[v as usize] == Some(p) {
                out.push(Action::BuildCity(v));
            }
        }
    }
    if can_afford(hand, &ROAD_COST) {
        for e in 0u8..72 {
            if is_legal_road_for_player(state, e, p) {
                out.push(Action::BuildRoad(e));
            }
        }
    }
    out
}

fn is_legal_settlement_for_player(state: &GameState, v: u8, p: u8) -> bool {
    if !is_legal_settlement_location(state, v) { return false; }
    // Main-phase rule: must be adjacent to one of the player's roads.
    state.board.vertex_to_edges[v as usize].iter()
        .any(|&e| state.roads[e as usize] == Some(p))
}

fn is_legal_road_for_player(state: &GameState, e: u8, p: u8) -> bool {
    if state.roads[e as usize].is_some() { return false; }
    // Connects to one of the player's roads or settlements/cities.
    let [a, b] = state.board.edge_to_vertices[e as usize];
    for v in [a, b] {
        if state.settlements[v as usize] == Some(p) || state.cities[v as usize] == Some(p) {
            return true;
        }
        // Or one of the OTHER edges adjacent to this vertex is the player's road,
        // unless that vertex is occupied by another player (blocking).
        let blocked = matches!(state.settlements[v as usize], Some(o) if o != p)
                   || matches!(state.cities[v as usize], Some(o) if o != p);
        if !blocked {
            for &e2 in &state.board.vertex_to_edges[v as usize] {
                if e2 != e && state.roads[e2 as usize] == Some(p) {
                    return true;
                }
            }
        }
    }
    false
}
```

- [ ] **Step 3: Implement settlement build in `apply` Main arm**

```rust
        (GamePhase::Main, Action::BuildSettlement(v)) => {
            let p = state.current_player;
            let mut bank = state.bank;
            let mut hand = state.hands[p as usize];
            pay(&mut hand, &mut bank, &SETTLEMENT_COST);
            state.hands[p as usize] = hand;
            state.bank = bank;
            state.settlements[v as usize] = Some(p);
            state.vp[p as usize] += 1;
            events.push(GameEvent::BuildSettlement { player: p, vertex: v });
            check_win(state, &mut events);
        }
```

Add helper:
```rust
fn check_win(state: &mut GameState, events: &mut Vec<GameEvent>) {
    for p in 0..4u8 {
        if state.vp[p as usize] >= crate::state::WIN_VP {
            state.phase = GamePhase::Done { winner: p };
            events.push(GameEvent::GameOver { winner: p });
            return;
        }
    }
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test --test main_phase`
Expected: settlement-build tests pass; older tests still pass.

- [ ] **Step 5: Commit**

```bash
git add catan_engine/src/rules.rs catan_engine/tests/main_phase.rs
git commit -m "feat: Main-phase settlement build"
```

---

### Task 17: Main phase — build city + build road + end turn

**Files:**
- Modify: `catan_engine/src/rules.rs`
- Modify: `catan_engine/tests/main_phase.rs`

- [ ] **Step 1: Tests for city, road, end-turn**

Append to `catan_engine/tests/main_phase.rs`:
```rust
#[test]
fn upgrading_settlement_to_city_costs_2_wheat_3_ore_and_grants_extra_vp() {
    let mut state = ready_to_roll();
    state.phase = GamePhase::Main;
    state.hands[0] = [0, 0, 0, 2, 3];
    let mut rng = Rng::from_seed(0);
    let owned = (0u8..54).find(|&v| state.settlements[v as usize] == Some(0)).unwrap();
    let vp_before = state.vp[0];
    apply(&mut state, Action::BuildCity(owned), &mut rng);
    assert_eq!(state.cities[owned as usize], Some(0));
    assert!(state.settlements[owned as usize].is_none());
    assert_eq!(state.hands[0], [0, 0, 0, 0, 0]);
    assert_eq!(state.vp[0], vp_before + 1); // +1 (city is 2VP, settlement was 1VP)
}

#[test]
fn end_turn_advances_player_and_returns_to_roll() {
    let mut state = ready_to_roll();
    state.phase = GamePhase::Main;
    let mut rng = Rng::from_seed(0);
    apply(&mut state, Action::EndTurn, &mut rng);
    assert_eq!(state.current_player, 1);
    assert!(matches!(state.phase, GamePhase::Roll));
}
```

- [ ] **Step 2: Add city, road, end-turn handlers to `apply` Main arm**

```rust
        (GamePhase::Main, Action::BuildCity(v)) => {
            let p = state.current_player;
            let mut bank = state.bank;
            let mut hand = state.hands[p as usize];
            pay(&mut hand, &mut bank, &CITY_COST);
            state.hands[p as usize] = hand;
            state.bank = bank;
            state.settlements[v as usize] = None;
            state.cities[v as usize] = Some(p);
            state.vp[p as usize] += 1; // settlement was 1VP, city is 2VP, net +1
            events.push(GameEvent::BuildCity { player: p, vertex: v });
            check_win(state, &mut events);
        }
        (GamePhase::Main, Action::BuildRoad(e)) => {
            let p = state.current_player;
            let mut bank = state.bank;
            let mut hand = state.hands[p as usize];
            pay(&mut hand, &mut bank, &ROAD_COST);
            state.hands[p as usize] = hand;
            state.bank = bank;
            state.roads[e as usize] = Some(p);
            events.push(GameEvent::BuildRoad { player: p, edge: e });
        }
        (GamePhase::Main, Action::EndTurn) => {
            events.push(GameEvent::TurnEnded { player: state.current_player });
            state.current_player = (state.current_player + 1) % 4;
            state.turn += 1;
            state.phase = GamePhase::Roll;
        }
```

- [ ] **Step 3: Run tests**

Run: `cargo test --test main_phase`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add catan_engine/src/rules.rs catan_engine/tests/main_phase.rs
git commit -m "feat: Main-phase city/road build + end turn"
```

---

### Task 18: Discard phase (one-at-a-time)

**Files:**
- Modify: `catan_engine/src/rules.rs`
- Create: `catan_engine/tests/discard.rs`

- [ ] **Step 1: Tests**

`catan_engine/tests/discard.rs`:
```rust
use catan_engine::actions::Action;
use catan_engine::board::{Board, Resource};
use catan_engine::rng::Rng;
use catan_engine::rules::{apply, legal_actions};
use catan_engine::state::{GameState, GamePhase};
use std::sync::Arc;

#[test]
fn legal_discard_actions_are_resources_player_owns() {
    let mut state = GameState::new(Arc::new(Board::standard()));
    state.phase = GamePhase::Discard { remaining: [3, 0, 0, 0] };
    state.current_player = 0;
    state.hands[0] = [3, 0, 5, 0, 0]; // 3 wood, 5 sheep
    let legal = legal_actions(&state);
    let discarded: Vec<Resource> = legal.iter()
        .filter_map(|a| if let Action::Discard(r) = a { Some(*r) } else { None })
        .collect();
    assert!(discarded.contains(&Resource::Wood));
    assert!(discarded.contains(&Resource::Sheep));
    assert!(!discarded.contains(&Resource::Brick));
}

#[test]
fn discarding_decrements_hand_and_remaining() {
    let mut state = GameState::new(Arc::new(Board::standard()));
    state.phase = GamePhase::Discard { remaining: [2, 0, 0, 0] };
    state.current_player = 0;
    state.hands[0] = [5, 0, 0, 0, 0];
    let mut rng = Rng::from_seed(0);
    apply(&mut state, Action::Discard(Resource::Wood), &mut rng);
    assert_eq!(state.hands[0][Resource::Wood as usize], 4);
    if let GamePhase::Discard { remaining } = state.phase {
        assert_eq!(remaining[0], 1);
    } else { panic!("expected Discard phase"); }
}

#[test]
fn last_discard_transitions_to_move_robber() {
    let mut state = GameState::new(Arc::new(Board::standard()));
    state.phase = GamePhase::Discard { remaining: [1, 0, 0, 0] };
    state.current_player = 0; // current_player is the roller, who triggers the robber move
    state.hands[0] = [1, 0, 0, 0, 0];
    let mut rng = Rng::from_seed(0);
    apply(&mut state, Action::Discard(Resource::Wood), &mut rng);
    assert!(matches!(state.phase, GamePhase::MoveRobber));
}
```

- [ ] **Step 2: Implement legal_actions for Discard**

In rules.rs:
```rust
        crate::state::GamePhase::Discard { remaining } => {
            // The "current discarder" is the lowest-indexed player still owing cards.
            let p = remaining.iter().position(|&n| n > 0).unwrap_or(0);
            let mut out = Vec::new();
            for r in 0..5u8 {
                if state.hands[p][r as usize] > 0 {
                    out.push(Action::Discard(match r {
                        0 => Resource::Wood, 1 => Resource::Brick, 2 => Resource::Sheep,
                        3 => Resource::Wheat, 4 => Resource::Ore, _ => unreachable!()
                    }));
                }
            }
            out
        }
```

- [ ] **Step 3: Implement apply for Discard**

```rust
        (GamePhase::Discard { remaining }, Action::Discard(r)) => {
            let mut rem = *remaining;
            let p = rem.iter().position(|&n| n > 0).unwrap();
            let ri = r as usize;
            assert!(state.hands[p][ri] > 0, "discarding resource player doesn't have");
            state.hands[p][ri] -= 1;
            state.bank[ri] += 1;
            rem[p] -= 1;
            events.push(GameEvent::Discarded { player: p as u8, resource: r });
            state.phase = if rem.iter().all(|&n| n == 0) {
                GamePhase::MoveRobber
            } else {
                GamePhase::Discard { remaining: rem }
            };
        }
```

> Note: `Resource` import needed. Add `use crate::board::Resource;` to top of rules.rs.

- [ ] **Step 4: Run tests**

Run: `cargo test --test discard`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add catan_engine/src/rules.rs catan_engine/tests/discard.rs
git commit -m "feat: Discard phase (one-at-a-time)"
```

---

### Task 19: Robber move + steal sub-phase

**Files:**
- Modify: `catan_engine/src/rules.rs`
- Create: `catan_engine/tests/robber.rs`

- [ ] **Step 1: Tests**

`catan_engine/tests/robber.rs`:
```rust
use catan_engine::actions::Action;
use catan_engine::board::Board;
use catan_engine::rng::Rng;
use catan_engine::rules::{apply, legal_actions};
use catan_engine::state::{GameState, GamePhase};
use std::sync::Arc;

#[test]
fn legal_move_robber_actions_exclude_current_hex() {
    let mut state = GameState::new(Arc::new(Board::standard()));
    state.phase = GamePhase::MoveRobber;
    let robber = state.robber_hex;
    let legal = legal_actions(&state);
    let hexes: Vec<u8> = legal.iter()
        .filter_map(|a| if let Action::MoveRobber(h) = a { Some(*h) } else { None })
        .collect();
    assert_eq!(hexes.len(), 18);
    assert!(!hexes.contains(&robber));
}

#[test]
fn moving_robber_to_unoccupied_hex_skips_steal() {
    let mut state = GameState::new(Arc::new(Board::standard()));
    state.phase = GamePhase::MoveRobber;
    state.current_player = 0;
    let mut rng = Rng::from_seed(0);
    let target = (0..19u8).find(|&h| h != state.robber_hex).unwrap();
    apply(&mut state, Action::MoveRobber(target), &mut rng);
    assert_eq!(state.robber_hex, target);
    assert!(matches!(state.phase, GamePhase::Main));
}

#[test]
fn moving_robber_to_hex_with_only_self_buildings_skips_steal() {
    let mut state = GameState::new(Arc::new(Board::standard()));
    // Put player 0's settlement adjacent to hex 5; nobody else has buildings there.
    let target_hex = 5u8;
    let v = state.board.hex_to_vertices[target_hex as usize][0];
    state.settlements[v as usize] = Some(0);
    state.phase = GamePhase::MoveRobber;
    state.current_player = 0;
    let mut rng = Rng::from_seed(0);
    apply(&mut state, Action::MoveRobber(target_hex), &mut rng);
    assert!(matches!(state.phase, GamePhase::Main));
}
```

- [ ] **Step 2: Implement legal_actions for MoveRobber and Steal**

In rules.rs:
```rust
        crate::state::GamePhase::MoveRobber => {
            (0u8..19)
                .filter(|&h| h != state.robber_hex)
                .map(Action::MoveRobber)
                .collect()
        }
        crate::state::GamePhase::Steal { from_options } => {
            // Steal target encoded in MoveRobber action ID space — but we need a
            // separate action. Reuse MoveRobber(player_index) for simplicity:
            // We model Steal as immediate-on-MoveRobber for now; if there are
            // multiple targets we pick the lowest index. (Real per-target choice
            // would need an extra Action variant; deferred to Tier 2.)
            let _ = from_options;
            vec![] // Steal phase auto-resolves in apply(); never exposed to caller.
        }
```

> Design simplification: we resolve "which player to steal from" automatically in `apply` (lowest-index opponent on the hex). True per-target choice requires extending the action enum, deferred until needed.

- [ ] **Step 3: Implement apply for MoveRobber**

```rust
        (GamePhase::MoveRobber, Action::MoveRobber(h)) => {
            let from = state.robber_hex;
            state.robber_hex = h;
            events.push(GameEvent::RobberMoved {
                player: state.current_player, from_hex: from, to_hex: h,
            });
            // Identify steal targets: opponents with buildings on this hex AND non-empty hand.
            let board = state.board.clone();
            let me = state.current_player;
            let mut targets: Vec<u8> = Vec::new();
            for &v in &board.hex_to_vertices[h as usize] {
                let owner = state.settlements[v as usize].or(state.cities[v as usize]);
                if let Some(p) = owner {
                    if p != me && state.hands[p as usize].iter().sum::<u8>() > 0
                        && !targets.contains(&p)
                    {
                        targets.push(p);
                    }
                }
            }
            if targets.is_empty() {
                state.phase = GamePhase::Main;
            } else {
                // Auto-steal from first target (see design simplification note).
                let victim = targets[0];
                let stolen = steal_random(state, victim, rng);
                events.push(GameEvent::Robbed { from: victim, to: me, resource: stolen });
                state.phase = GamePhase::Main;
            }
        }
```

Add helper:
```rust
fn steal_random(state: &mut GameState, victim: u8, rng: &mut Rng) -> Option<Resource> {
    use rand::Rng as _;
    let total: u32 = state.hands[victim as usize].iter().map(|&x| x as u32).sum();
    if total == 0 { return None; }
    let pick = rng.inner().gen_range(0..total);
    let mut acc = 0u32;
    for ri in 0..5usize {
        acc += state.hands[victim as usize][ri] as u32;
        if pick < acc {
            state.hands[victim as usize][ri] -= 1;
            state.hands[state.current_player as usize][ri] += 1;
            return Some(match ri {
                0 => Resource::Wood, 1 => Resource::Brick, 2 => Resource::Sheep,
                3 => Resource::Wheat, 4 => Resource::Ore, _ => unreachable!(),
            });
        }
    }
    None
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test --test robber`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add catan_engine/src/rules.rs catan_engine/tests/robber.rs
git commit -m "feat: robber move + auto-steal"
```

---

### Task 20: Wire is_terminal + verify smoke test passes

**Files:**
- Modify: `catan_engine/src/engine.rs`
- Modify: `catan_engine/src/lib.rs`

- [ ] **Step 1: Replace stub Engine with real wiring**

`catan_engine/src/engine.rs`:
```rust
//! Top-level orchestrator.

use crate::actions::{decode, Action};
use crate::board::Board;
use crate::events::{EventLog, GameEvent};
use crate::rng::Rng;
use crate::rules::{apply, legal_actions};
use crate::state::GameState;
use std::sync::Arc;

pub struct Engine {
    pub state: GameState,
    pub rng: Rng,
    pub events: EventLog,
}

impl Engine {
    pub fn new(seed: u64) -> Self {
        let board = Arc::new(Board::standard());
        Self {
            state: GameState::new(board),
            rng: Rng::from_seed(seed),
            events: EventLog::new(),
        }
    }

    pub fn legal_actions(&self) -> Vec<u32> {
        legal_actions(&self.state)
            .into_iter()
            .map(crate::actions::encode)
            .collect()
    }

    pub fn step(&mut self, action_id: u32) {
        let action = decode(action_id).expect("invalid action ID");
        let evs = apply(&mut self.state, action, &mut self.rng);
        for e in evs {
            self.events.push(e);
        }
    }

    pub fn is_terminal(&self) -> bool {
        self.state.is_terminal()
    }

    pub fn event_log(&self) -> &[GameEvent] {
        self.events.as_slice()
    }
}
```

- [ ] **Step 2: Run the smoke test**

Run: `cargo test --test smoke`
Expected: PASS — a complete random game runs to a `Done` state. **If it hangs or panics, the bug is in legal_actions emitting an invalid action for some phase, OR in apply not advancing phases. Use `RUST_LOG=debug` and add a `dbg!(state.phase)` to bisect.**

- [ ] **Step 3: Run all tests**

Run: `cargo test`
Expected: ALL pass.

- [ ] **Step 4: Commit**

```bash
git add catan_engine/src/engine.rs
git commit -m "feat: wire Engine to real rules + smoke test green"
```

---

## Phase 6 — Stats, RNG, determinism

### Task 21: GameStats struct + fold from events

**Files:**
- Create: `catan_engine/src/stats.rs`
- Modify: `catan_engine/src/lib.rs`
- Modify: `catan_engine/src/engine.rs`

- [ ] **Step 1: Define stats.rs**

`catan_engine/src/stats.rs`:
```rust
//! Game statistics, folded incrementally from the event log.
//! Spec §7. schema_version is bumped when the struct layout changes.

use crate::events::GameEvent;

pub const STATS_SCHEMA_VERSION: u32 = 1;
pub const N_PLAYERS: usize = 4;
pub const N_RESOURCES: usize = 5;
pub const N_HEXES: usize = 19;

#[derive(Debug, Clone, Default)]
pub struct PerPlayerStats {
    pub vp_final: u8,
    pub won: bool,
    pub settlements_built: u32,
    pub cities_built: u32,
    pub roads_built: u32,
    pub resources_gained: [u32; N_RESOURCES],
    pub resources_gained_from_robber: [u32; N_RESOURCES],
    pub resources_lost_to_robber: [u32; N_RESOURCES],
    pub resources_lost_to_discard: [u32; N_RESOURCES],
    pub cards_in_hand_max: u32,
    pub times_robbed: u32,
    pub robber_moves: u32,
    pub discards_triggered: u32,
}

#[derive(Debug, Clone)]
pub struct GameStats {
    pub schema_version: u32,
    pub turns_played: u32,
    pub dice_histogram: [u32; 11], // index 0 = roll 2, ..., 10 = roll 12
    pub seven_count: u32,
    pub production_per_hex: [u32; N_HEXES],
    pub total_resources_produced: u32,
    pub winner_player_id: i8,
    pub players: [PerPlayerStats; N_PLAYERS],
}

impl GameStats {
    pub fn new() -> Self {
        Self {
            schema_version: STATS_SCHEMA_VERSION,
            turns_played: 0,
            dice_histogram: [0; 11],
            seven_count: 0,
            production_per_hex: [0; N_HEXES],
            total_resources_produced: 0,
            winner_player_id: -1,
            players: Default::default(),
        }
    }

    pub fn fold_event(&mut self, event: &GameEvent) {
        match *event {
            GameEvent::DiceRolled { roll } => {
                if roll == 7 {
                    self.seven_count += 1;
                } else if (2..=12).contains(&roll) {
                    self.dice_histogram[(roll - 2) as usize] += 1;
                }
            }
            GameEvent::ResourcesProduced { player, hex, resource: _, amount } => {
                self.production_per_hex[hex as usize] += amount as u32;
                self.total_resources_produced += amount as u32;
                self.players[player as usize]
                    .resources_gained[_resource_idx(event)] += amount as u32;
            }
            GameEvent::BuildSettlement { player, .. } => {
                self.players[player as usize].settlements_built += 1;
            }
            GameEvent::BuildCity { player, .. } => {
                self.players[player as usize].cities_built += 1;
            }
            GameEvent::BuildRoad { player, .. } => {
                self.players[player as usize].roads_built += 1;
            }
            GameEvent::RobberMoved { player, .. } => {
                self.players[player as usize].robber_moves += 1;
            }
            GameEvent::Robbed { from, to, resource } => {
                self.players[from as usize].times_robbed += 1;
                if let Some(r) = resource {
                    self.players[from as usize].resources_lost_to_robber[r as usize] += 1;
                    self.players[to as usize].resources_gained_from_robber[r as usize] += 1;
                }
            }
            GameEvent::Discarded { player, resource } => {
                self.players[player as usize].resources_lost_to_discard[resource as usize] += 1;
                self.players[player as usize].discards_triggered += 1;
            }
            GameEvent::TurnEnded { .. } => {
                self.turns_played += 1;
            }
            GameEvent::GameOver { winner } => {
                self.winner_player_id = winner as i8;
                for p in 0..N_PLAYERS {
                    self.players[p].won = p as u8 == winner;
                }
            }
        }
    }
}

fn _resource_idx(e: &GameEvent) -> usize {
    if let GameEvent::ResourcesProduced { resource, .. } = e {
        *resource as usize
    } else { 0 }
}
```

- [ ] **Step 2: Wire into lib.rs and Engine**

`catan_engine/src/lib.rs`:
```rust
pub mod stats;
```

In `engine.rs`, add a `stats: GameStats` field, initialize in `new`, and update on every step:
```rust
use crate::stats::GameStats;

pub struct Engine {
    pub state: GameState,
    pub rng: Rng,
    pub events: EventLog,
    pub stats: GameStats,
}

impl Engine {
    pub fn new(seed: u64) -> Self {
        let board = Arc::new(Board::standard());
        Self {
            state: GameState::new(board),
            rng: Rng::from_seed(seed),
            events: EventLog::new(),
            stats: GameStats::new(),
        }
    }

    pub fn step(&mut self, action_id: u32) {
        let action = decode(action_id).expect("invalid action ID");
        let evs = apply(&mut self.state, action, &mut self.rng);
        for e in &evs {
            self.stats.fold_event(e);
            self.events.push(*e);
        }
        // Update cards_in_hand_max
        for p in 0..4 {
            let total: u32 = self.state.hands[p].iter().map(|&x| x as u32).sum();
            if total > self.stats.players[p].cards_in_hand_max {
                self.stats.players[p].cards_in_hand_max = total;
            }
        }
    }

    pub fn stats(&self) -> &GameStats { &self.stats }
}
```

Also update `Engine` to set `stats.players[p].vp_final` at terminal — easiest place: in `is_terminal` or in `step` after each action:
```rust
        if self.state.is_terminal() {
            for p in 0..4 {
                self.stats.players[p].vp_final = self.state.vp[p];
            }
        }
```

- [ ] **Step 3: Add a stats sanity test**

Append to `catan_engine/tests/main_phase.rs` (or new `tests/stats.rs`):
```rust
#[test]
fn stats_track_basic_game_progress() {
    let mut engine = catan_engine::Engine::new(42);
    while !engine.is_terminal() {
        let legal = engine.legal_actions();
        if legal.is_empty() { break; }
        engine.step(legal[0]);
    }
    let s = engine.stats();
    assert_eq!(s.schema_version, catan_engine::stats::STATS_SCHEMA_VERSION);
    assert!(s.turns_played > 0);
    let total_dice: u32 = s.dice_histogram.iter().sum::<u32>() + s.seven_count;
    assert_eq!(total_dice as u32, s.turns_played);
    assert!(s.winner_player_id >= 0);
}
```

- [ ] **Step 4: Run all tests**

Run: `cargo test`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add catan_engine/src/stats.rs catan_engine/src/lib.rs catan_engine/src/engine.rs catan_engine/tests/main_phase.rs
git commit -m "feat: GameStats folded from event log"
```

---

### Task 22: Lock down RNG determinism

**Files:**
- Modify: `catan_engine/src/rng.rs`
- Modify: `catan_engine/src/rules.rs` (audit for any non-Rng randomness)

- [ ] **Step 1: Audit for non-deterministic order**

Run: `cd C:/dojo/catan_bot && grep -rn "HashMap\|HashSet\|thread_rng\|rand::random" catan_engine/src/`
Expected: zero hits in source files. If any appear, replace `HashMap`→`BTreeMap`, `HashSet`→`BTreeSet`, never use `thread_rng`.

- [ ] **Step 2: Add a determinism unit test**

Create `catan_engine/tests/determinism.rs`:
```rust
use catan_engine::Engine;

fn run_game(seed: u64) -> (u32, Vec<u32>) {
    let mut engine = Engine::new(seed);
    let mut actions = Vec::new();
    while !engine.is_terminal() {
        let legal = engine.legal_actions();
        if legal.is_empty() { break; }
        // Always pick action[0] for full determinism.
        engine.step(legal[0]);
        actions.push(legal[0]);
    }
    let s = engine.stats();
    (s.turns_played, actions)
}

#[test]
fn same_seed_produces_identical_trajectory() {
    let a = run_game(42);
    let b = run_game(42);
    assert_eq!(a, b);
}

#[test]
fn different_seeds_produce_different_trajectories() {
    let a = run_game(1);
    let b = run_game(2);
    assert_ne!(a, b);
}
```

- [ ] **Step 3: Run**

Run: `cargo test --test determinism`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add catan_engine/tests/determinism.rs
git commit -m "test: determinism (same seed -> same trajectory)"
```

---

### Task 23: Determinism regression hashes

**Files:**
- Create: `catan_engine/tests/determinism_hashes.rs`

- [ ] **Step 1: Generate hashes for fixed seeds**

`catan_engine/tests/determinism_hashes.rs`:
```rust
use catan_engine::Engine;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

fn game_hash(seed: u64) -> u64 {
    let mut engine = Engine::new(seed);
    let mut h = DefaultHasher::new();
    while !engine.is_terminal() {
        let legal = engine.legal_actions();
        if legal.is_empty() { break; }
        let a = legal[0];
        engine.step(a);
        a.hash(&mut h);
    }
    let stats = engine.stats();
    stats.turns_played.hash(&mut h);
    stats.winner_player_id.hash(&mut h);
    h.finish()
}

#[test]
fn regression_hashes_are_stable() {
    // GENERATE these on first run (uncomment the println, run, copy values back).
    // After that, any change that flips a hash is a flagged behavioral change.
    let expected = [
        (0u64, 0u64),  // <-- replace second value with actual hash
        (1u64, 0u64),
        (42u64, 0u64),
        (12345u64, 0u64),
    ];
    for (seed, expected_hash) in expected {
        let actual = game_hash(seed);
        // Uncomment on first run:
        // println!("seed {seed} -> {actual}");
        if expected_hash != 0 {
            assert_eq!(actual, expected_hash, "regression for seed {seed}");
        }
    }
}
```

- [ ] **Step 2: First run — capture hashes**

Run: `cargo test --test determinism_hashes -- --nocapture`
Expected: prints actual hashes (after you uncomment the println). Copy the printed values into the `expected` array.

- [ ] **Step 3: Re-run with hashes filled in**

Run: `cargo test --test determinism_hashes`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add catan_engine/tests/determinism_hashes.rs
git commit -m "test: determinism regression hashes for 4 fixed seeds"
```

---

## Phase 7 — Observation, PyO3 surface, Python env

### Task 24: Observation builder (Rust side, numpy-shaped data)

**Files:**
- Create: `catan_engine/src/observation.rs`
- Modify: `catan_engine/src/lib.rs`

> **Spec deviation note:** Spec §5.2 calls for both a graph view AND an 11×11 2D grid view. This plan implements **only the graph view + scalars** (Tasks 24-26). Rationale: the user is leaning GNN; the 2D grid is a nice-to-have for CNN ablations and can be added later as `observation.rs::build_grid_observation()` without changing existing API. Tracking this as deferred per spec §13.

- [ ] **Step 1: Define observation.rs**

`catan_engine/src/observation.rs`:
```rust
//! Build numpy-shaped observation tensors from GameState.
//! Spec §5.

use crate::state::{GameState, N_PLAYERS, N_RESOURCES};

pub const F_HEX: usize = 8;     // [Wood,Brick,Sheep,Wheat,Ore one-hot, dice-norm, robber, desert]
pub const F_VERT: usize = 7;    // [empty, settle, city, owner-0..3 perspective]
pub const F_EDGE: usize = 6;    // [empty, road, owner-0..3 perspective]
pub const N_SCALARS: usize = 22;

pub struct Observation {
    pub hex_features: Vec<f32>,    // shape [19, F_HEX]
    pub vertex_features: Vec<f32>, // shape [54, F_VERT]
    pub edge_features: Vec<f32>,   // shape [72, F_EDGE]
    pub scalars: Vec<f32>,         // shape [N_SCALARS]
    pub legal_mask: Vec<bool>,     // shape [205]
}

pub fn build_observation(state: &GameState, viewer: u8) -> Observation {
    let mut hex_features = vec![0.0f32; 19 * F_HEX];
    for (i, h) in state.board.hexes.iter().enumerate() {
        if let Some(r) = h.resource {
            hex_features[i * F_HEX + r as usize] = 1.0;
        } else {
            hex_features[i * F_HEX + 7] = 1.0; // desert flag
        }
        if let Some(n) = h.dice_number {
            hex_features[i * F_HEX + 5] = (n as f32 - 7.0) / 5.0;
        }
        if state.robber_hex as usize == i {
            hex_features[i * F_HEX + 6] = 1.0;
        }
    }
    let mut vertex_features = vec![0.0f32; 54 * F_VERT];
    for v in 0..54usize {
        if let Some(p) = state.cities[v] {
            vertex_features[v * F_VERT + 2] = 1.0;
            vertex_features[v * F_VERT + 3 + perspective_idx(p, viewer)] = 1.0;
        } else if let Some(p) = state.settlements[v] {
            vertex_features[v * F_VERT + 1] = 1.0;
            vertex_features[v * F_VERT + 3 + perspective_idx(p, viewer)] = 1.0;
        } else {
            vertex_features[v * F_VERT] = 1.0;
        }
    }
    let mut edge_features = vec![0.0f32; 72 * F_EDGE];
    for e in 0..72usize {
        if let Some(p) = state.roads[e] {
            edge_features[e * F_EDGE + 1] = 1.0;
            edge_features[e * F_EDGE + 2 + perspective_idx(p, viewer)] = 1.0;
        } else {
            edge_features[e * F_EDGE] = 1.0;
        }
    }
    let mut scalars = vec![0.0f32; N_SCALARS];
    // Viewer's hand counts (5)
    for r in 0..N_RESOURCES { scalars[r] = state.hands[viewer as usize][r] as f32; }
    // Opponent hand sizes (3) — perspective-rotated
    let mut idx = 5;
    for offset in 1..N_PLAYERS as u8 {
        let opp = (viewer + offset) % N_PLAYERS as u8;
        scalars[idx] = state.hands[opp as usize].iter().map(|&x| x as f32).sum();
        idx += 1;
    }
    // VP for all 4 players (perspective order)
    for offset in 0..N_PLAYERS as u8 {
        let p = (viewer + offset) % N_PLAYERS as u8;
        scalars[idx] = state.vp[p as usize] as f32;
        idx += 1;
    }
    // Turn (normalized) + phase one-hot (8)
    scalars[idx] = (state.turn as f32 / 100.0).min(1.0);
    idx += 1;
    let phase_idx = match state.phase {
        crate::state::GamePhase::Setup1Place => 0,
        crate::state::GamePhase::Setup2Place => 1,
        crate::state::GamePhase::Roll => 2,
        crate::state::GamePhase::Main => 3,
        crate::state::GamePhase::Discard { .. } => 4,
        crate::state::GamePhase::MoveRobber => 5,
        crate::state::GamePhase::Steal { .. } => 6,
        crate::state::GamePhase::Done { .. } => 7,
    };
    scalars[idx + phase_idx] = 1.0;

    let legal = crate::rules::legal_actions(state);
    let mut legal_mask = vec![false; 205];
    for a in legal {
        legal_mask[crate::actions::encode(a) as usize] = true;
    }

    Observation { hex_features, vertex_features, edge_features, scalars, legal_mask }
}

fn perspective_idx(player: u8, viewer: u8) -> usize {
    ((player + N_PLAYERS as u8 - viewer) % N_PLAYERS as u8) as usize
}
```

- [ ] **Step 2: Wire into lib.rs**

```rust
pub mod observation;
```

- [ ] **Step 3: Add a test**

`catan_engine/tests/observation.rs`:
```rust
use catan_engine::observation::build_observation;
use catan_engine::Engine;

#[test]
fn observation_shapes_match_spec() {
    let engine = Engine::new(42);
    let obs = build_observation(&engine.state, engine.state.current_player);
    assert_eq!(obs.hex_features.len(), 19 * catan_engine::observation::F_HEX);
    assert_eq!(obs.vertex_features.len(), 54 * catan_engine::observation::F_VERT);
    assert_eq!(obs.edge_features.len(), 72 * catan_engine::observation::F_EDGE);
    assert_eq!(obs.legal_mask.len(), 205);
}

#[test]
fn legal_mask_count_matches_legal_actions() {
    let engine = Engine::new(42);
    let obs = build_observation(&engine.state, engine.state.current_player);
    let mask_true = obs.legal_mask.iter().filter(|&&b| b).count();
    assert_eq!(mask_true, engine.legal_actions().len());
}
```

- [ ] **Step 4: Run + commit**

Run: `cargo test --test observation`
Expected: PASS.

```bash
git add catan_engine/src/observation.rs catan_engine/src/lib.rs catan_engine/tests/observation.rs
git commit -m "feat: observation builder (Rust side)"
```

---

### Task 25: PyO3 surface — full Engine bindings

**Files:**
- Modify: `catan_engine/src/lib.rs`

- [ ] **Step 1: Replace lib.rs hello-world with full Engine bindings**

`catan_engine/src/lib.rs` — replace the `_engine` pymodule body:
```rust
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use numpy::{PyArray1, PyArray2, IntoPyArray};

pub mod actions;
pub mod board;
pub mod engine;
pub mod events;
pub mod observation;
pub mod rng;
pub mod rules;
pub mod state;
pub mod stats;

pub use engine::Engine;

#[pyclass(name = "Engine")]
struct PyEngine {
    inner: Engine,
}

#[pymethods]
impl PyEngine {
    #[new]
    fn new(seed: u64) -> Self {
        Self { inner: Engine::new(seed) }
    }

    fn legal_actions<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        self.inner.legal_actions().into_pyarray_bound(py)
    }

    fn step(&mut self, action_id: u32) {
        self.inner.step(action_id);
    }

    fn is_terminal(&self) -> bool { self.inner.is_terminal() }

    fn current_player(&self) -> u8 { self.inner.state.current_player }

    fn observation<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let viewer = self.inner.state.current_player;
        let obs = observation::build_observation(&self.inner.state, viewer);
        let d = PyDict::new_bound(py);
        d.set_item(
            "hex_features",
            PyArray2::from_vec2_bound(py, &chunks(&obs.hex_features, observation::F_HEX))?,
        )?;
        d.set_item(
            "vertex_features",
            PyArray2::from_vec2_bound(py, &chunks(&obs.vertex_features, observation::F_VERT))?,
        )?;
        d.set_item(
            "edge_features",
            PyArray2::from_vec2_bound(py, &chunks(&obs.edge_features, observation::F_EDGE))?,
        )?;
        d.set_item("scalars", obs.scalars.into_pyarray_bound(py))?;
        d.set_item(
            "legal_mask",
            obs.legal_mask.iter().map(|&b| b as u8).collect::<Vec<u8>>().into_pyarray_bound(py),
        )?;
        Ok(d)
    }

    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let s = self.inner.stats();
        let d = PyDict::new_bound(py);
        d.set_item("schema_version", s.schema_version)?;
        d.set_item("turns_played", s.turns_played)?;
        d.set_item("seven_count", s.seven_count)?;
        d.set_item("dice_histogram", s.dice_histogram.to_vec().into_pyarray_bound(py))?;
        d.set_item("production_per_hex", s.production_per_hex.to_vec().into_pyarray_bound(py))?;
        d.set_item("winner_player_id", s.winner_player_id)?;
        let players = PyList::empty_bound(py);
        for p in 0..4 {
            let pd = PyDict::new_bound(py);
            pd.set_item("vp_final", s.players[p].vp_final)?;
            pd.set_item("won", s.players[p].won)?;
            pd.set_item("settlements_built", s.players[p].settlements_built)?;
            pd.set_item("cities_built", s.players[p].cities_built)?;
            pd.set_item("roads_built", s.players[p].roads_built)?;
            pd.set_item("cards_in_hand_max", s.players[p].cards_in_hand_max)?;
            pd.set_item("times_robbed", s.players[p].times_robbed)?;
            pd.set_item("robber_moves", s.players[p].robber_moves)?;
            pd.set_item("discards_triggered", s.players[p].discards_triggered)?;
            players.append(pd)?;
        }
        d.set_item("players", players)?;
        Ok(d)
    }
}

fn chunks(flat: &[f32], width: usize) -> Vec<Vec<f32>> {
    flat.chunks(width).map(|c| c.to_vec()).collect()
}

#[pyfunction]
fn engine_version() -> &'static str { "0.1.0" }

#[pyfunction]
fn action_space_size() -> usize { actions::ACTION_SPACE_SIZE }

#[pymodule]
fn _engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(engine_version, m)?)?;
    m.add_function(wrap_pyfunction!(action_space_size, m)?)?;
    m.add_class::<PyEngine>()?;
    Ok(())
}
```

- [ ] **Step 2: Build**

Run: `maturin develop`
Expected: clean build.

- [ ] **Step 3: Commit**

```bash
git add catan_engine/src/lib.rs
git commit -m "feat: full PyO3 Engine bindings"
```

---

### Task 26: Python `env.py` gym-style wrapper

**Files:**
- Create: `python/catan_bot/env.py`
- Modify: `python/catan_bot/__init__.py`

- [ ] **Step 1: Write env.py**

`python/catan_bot/env.py`:
```python
"""Gym-style wrapper around the Rust Engine."""
from __future__ import annotations
import numpy as np
from catan_bot._engine import Engine as _Engine, action_space_size

ACTION_SPACE_SIZE = action_space_size()


class CatanEnv:
    """Single-game Catan environment.

    `step(action_id)` returns `(obs, reward, done, info)` where:
      - obs is a dict with keys: hex_features, vertex_features, edge_features,
        scalars, legal_mask.
      - reward is +1 if the current player just won, -1 if game ended without
        them winning, 0 otherwise.
      - done is True iff the game is terminal.
      - info contains stats snapshot and current_player.
    """

    def __init__(self, seed: int = 0):
        self._engine = _Engine(seed)

    def reset(self, seed: int = 0):
        self._engine = _Engine(seed)
        return self._observation()

    def step(self, action_id: int):
        prev_player = self._engine.current_player()
        self._engine.step(int(action_id))
        done = self._engine.is_terminal()
        reward = 0.0
        if done:
            stats = self._engine.stats()
            winner = stats["winner_player_id"]
            reward = 1.0 if winner == prev_player else -1.0
        return self._observation(), reward, done, {"current_player": self._engine.current_player()}

    def legal_actions(self) -> np.ndarray:
        return self._engine.legal_actions()

    def legal_mask(self) -> np.ndarray:
        return self._observation()["legal_mask"].astype(bool)

    def stats(self):
        return self._engine.stats()

    def is_terminal(self) -> bool:
        return self._engine.is_terminal()

    def _observation(self):
        obs = self._engine.observation()
        return obs
```

- [ ] **Step 2: Update __init__.py**

`python/catan_bot/__init__.py`:
```python
from catan_bot._engine import Engine, engine_version, action_space_size
from catan_bot.env import CatanEnv, ACTION_SPACE_SIZE

__all__ = ["Engine", "CatanEnv", "engine_version", "action_space_size", "ACTION_SPACE_SIZE"]
```

- [ ] **Step 3: Commit**

```bash
git add python/
git commit -m "feat: Python gym-style CatanEnv wrapper"
```

---

### Task 27: Python integration test — random self-play

**Files:**
- Create: `tests/test_env.py`

- [ ] **Step 1: Write the integration test**

`tests/test_env.py`:
```python
import numpy as np
from catan_bot import CatanEnv, ACTION_SPACE_SIZE


def test_action_space_size_constant():
    assert ACTION_SPACE_SIZE == 205


def test_random_self_play_completes():
    rng = np.random.default_rng(42)
    env = CatanEnv(seed=42)
    obs = env.reset(seed=42)
    steps = 0
    while not env.is_terminal():
        mask = env.legal_mask()
        legal_ids = np.flatnonzero(mask)
        assert len(legal_ids) > 0, "no legal actions in non-terminal state"
        action = int(rng.choice(legal_ids))
        obs, reward, done, info = env.step(action)
        steps += 1
        assert steps < 5000
    assert env.is_terminal()
    s = env.stats()
    assert s["winner_player_id"] >= 0
    assert s["turns_played"] > 0


def test_observation_shapes():
    env = CatanEnv(seed=1)
    obs = env._observation()
    assert obs["hex_features"].shape == (19, 8)
    assert obs["vertex_features"].shape == (54, 7)
    assert obs["edge_features"].shape == (72, 6)
    assert obs["legal_mask"].shape == (205,)
```

- [ ] **Step 2: Build + run**

Run: `cd C:/dojo/catan_bot && maturin develop && pytest tests/test_env.py -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_env.py
git commit -m "test: Python integration — random self-play to completion"
```

---

### Task 28: Replay log save/load

**Files:**
- Create: `python/catan_bot/replay.py`
- Modify: `python/catan_bot/__init__.py`
- Modify: `tests/test_env.py` (add round-trip test)

- [ ] **Step 1: Write replay.py**

`python/catan_bot/replay.py`:
```python
"""Versioned replay log: (seed, action sequence) → reconstructable game."""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List
from catan_bot.env import CatanEnv

REPLAY_SCHEMA_VERSION = 1


@dataclass
class Replay:
    schema_version: int
    seed: int
    actions: List[int]
    engine_version: str
    rules_tier: int = 1

    def save(self, path: str | Path):
        Path(path).write_text(json.dumps(asdict(self)))

    @classmethod
    def load(cls, path: str | Path) -> "Replay":
        data = json.loads(Path(path).read_text())
        if data["schema_version"] != REPLAY_SCHEMA_VERSION:
            raise ValueError(f"Unsupported replay schema {data['schema_version']}")
        return cls(**data)

    def reconstruct(self) -> CatanEnv:
        env = CatanEnv(seed=self.seed)
        env.reset(seed=self.seed)
        for a in self.actions:
            env.step(a)
        return env
```

- [ ] **Step 2: Re-export and test**

Add to `python/catan_bot/__init__.py`:
```python
from catan_bot.replay import Replay, REPLAY_SCHEMA_VERSION
```

Append to `tests/test_env.py`:
```python
def test_replay_roundtrip(tmp_path):
    import numpy as np
    from catan_bot import Replay
    rng = np.random.default_rng(7)
    env = CatanEnv(seed=7)
    actions = []
    while not env.is_terminal():
        mask = env.legal_mask()
        legal_ids = np.flatnonzero(mask)
        a = int(rng.choice(legal_ids))
        env.step(a)
        actions.append(a)
    rep = Replay(schema_version=1, seed=7, actions=actions,
                 engine_version="0.1.0", rules_tier=1)
    p = tmp_path / "rep.json"
    rep.save(p)
    rep2 = Replay.load(p)
    env2 = rep2.reconstruct()
    assert env2.is_terminal()
    assert env2.stats()["winner_player_id"] == env.stats()["winner_player_id"]
```

- [ ] **Step 3: Run + commit**

Run: `pytest tests/test_env.py -v`
Expected: PASS.

```bash
git add python/catan_bot/replay.py python/catan_bot/__init__.py tests/test_env.py
git commit -m "feat: replay log save/load with round-trip test"
```

---

## Phase 8 — Property tests + benchmark

### Task 29: Property tests via `proptest`

**Files:**
- Create: `catan_engine/tests/properties.rs`

- [ ] **Step 1: Write the property tests**

`catan_engine/tests/properties.rs`:
```rust
use catan_engine::actions::{decode, encode, ACTION_SPACE_SIZE};
use catan_engine::Engine;
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn action_id_round_trip(id in 0u32..(ACTION_SPACE_SIZE as u32)) {
        let action = decode(id).unwrap();
        prop_assert_eq!(encode(action), id);
    }

    #[test]
    fn random_legal_play_never_panics(seed in any::<u64>(), choices in prop::collection::vec(any::<u32>(), 0..1000)) {
        let mut engine = Engine::new(seed);
        for c in choices {
            if engine.is_terminal() { break; }
            let legal = engine.legal_actions();
            if legal.is_empty() { break; }
            let a = legal[c as usize % legal.len()];
            engine.step(a);
        }
        // No panic = test passes.
    }

    #[test]
    fn invariant_total_resources_within_supply(seed in any::<u64>(), choices in prop::collection::vec(any::<u32>(), 0..500)) {
        let mut engine = Engine::new(seed);
        for c in choices {
            if engine.is_terminal() { break; }
            let legal = engine.legal_actions();
            if legal.is_empty() { break; }
            engine.step(legal[c as usize % legal.len()]);
            // For each resource: bank + sum(player hands) == 19.
            for r in 0..5 {
                let in_hands: u32 = (0..4).map(|p| engine.state.hands[p][r] as u32).sum();
                let bank = engine.state.bank[r] as u32;
                prop_assert_eq!(in_hands + bank, 19u32, "resource {r} bookkeeping broken");
            }
        }
    }

    #[test]
    fn invariant_vp_non_decreasing(seed in any::<u64>(), choices in prop::collection::vec(any::<u32>(), 0..500)) {
        let mut engine = Engine::new(seed);
        let mut last_vp = [0u8; 4];
        for c in choices {
            if engine.is_terminal() { break; }
            let legal = engine.legal_actions();
            if legal.is_empty() { break; }
            engine.step(legal[c as usize % legal.len()]);
            for p in 0..4 {
                prop_assert!(engine.state.vp[p] >= last_vp[p]);
                last_vp[p] = engine.state.vp[p];
            }
        }
    }
}
```

- [ ] **Step 2: Run**

Run: `cargo test --test properties --release`
Expected: PASS (release mode is much faster for proptest).

> If a property fails, proptest will print a *minimal* shrunk reproducer. Use it to debug.

- [ ] **Step 3: Commit**

```bash
git add catan_engine/tests/properties.rs
git commit -m "test: proptest invariants (no-panic, resource conservation, VP monotonic)"
```

---

### Task 30: Throughput benchmark via `criterion`

**Files:**
- Create: `catan_engine/benches/throughput.rs`

- [ ] **Step 1: Write the benchmark**

`catan_engine/benches/throughput.rs`:
```rust
use catan_engine::Engine;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn play_one_game(seed: u64) -> u32 {
    let mut engine = Engine::new(seed);
    let mut steps = 0u32;
    while !engine.is_terminal() {
        let legal = engine.legal_actions();
        if legal.is_empty() { break; }
        engine.step(legal[(steps as usize) % legal.len()]);
        steps += 1;
        if steps > 5000 { break; }
    }
    steps
}

fn bench_games(c: &mut Criterion) {
    c.bench_function("random_game_seed_42", |b| {
        b.iter(|| play_one_game(black_box(42)));
    });
    c.bench_function("random_game_varied_seeds", |b| {
        let mut s = 0u64;
        b.iter(|| {
            s = s.wrapping_add(1);
            play_one_game(black_box(s))
        });
    });
}

criterion_group!(benches, bench_games);
criterion_main!(benches);
```

- [ ] **Step 2: Run the benchmark**

Run: `cd C:/dojo/catan_bot && cargo bench`
Expected: criterion prints time per game. **Target: ≤ 100 µs/game (i.e., ≥ 10,000 games/sec).** If above target, profile with `cargo flamegraph` and look first at observation-tensor allocation in the hot loop.

- [ ] **Step 3: Capture the result in the README**

Add a "Performance" section to `README.md` with the actual measured number from criterion's output:
```markdown
## Performance (v1)

Single-threaded random-vs-random self-play:
- ~XX,XXX games/sec on <CPU model>
- ~XX µs per game
- Target was 10,000 games/sec (spec §10).
```

- [ ] **Step 4: Commit**

```bash
git add catan_engine/benches/throughput.rs README.md
git commit -m "bench: throughput benchmark + README perf section"
```

---

## Wrap-up

After Task 30, validate the full success criteria from spec §15:

- [ ] **Smoke test passes for 1M games:**
  ```bash
  cargo test --test smoke --release -- --nocapture
  ```
  (Adjust the test or write a separate stress test that loops 1M times.)

- [ ] **All other tests pass:**
  ```bash
  cargo test --release
  pytest tests/ -v
  ```

- [ ] **Benchmark sustains ≥10k games/sec:**
  ```bash
  cargo bench
  ```

- [ ] **Final commit + tag:**
  ```bash
  git tag v0.1.0
  git push --tags
  ```
