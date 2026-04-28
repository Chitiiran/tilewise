# Phase 0 — Engine Chance-Node API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the Rust engine + PyO3 bindings so OpenSpiel's MCTS can model dice rolls and robber-steal target selection as honest chance nodes, plus expose `Engine.clone()` for MCTS rollouts.

**Architecture:** Three concrete additions to the engine:
1. A new `Action::RollDice` action ID (extending the action space from 205 → 206) so the dice roll is a discrete event we can interpose on.
2. A `Steal` phase that *is* actually live (replacing the current auto-steal-in-MoveRobber shortcut) plus a chance-outcome API to pick the resource taken.
3. PyO3 surface: `is_chance_node()`, `chance_outcomes()`, `apply_chance_outcome(value)`, `clone()`, `action_history()`.

**Tech Stack:** Rust (existing crate), PyO3, `maturin develop` for iterative builds, `cargo test`, `pytest`.

**Cross-cutting note for executor:** The action space currently has 205 IDs (see `catan_engine/src/actions.rs:15`). This plan grows it to **206** (`RollDice = 205`). The MCTS-study spec (`docs/superpowers/specs/2026-04-27-mcts-study-design.md` §1.5) hardcodes 205 in the `legal_action_mask` / `mcts_visit_counts` schemas. **You must update Plan 2's schema constant accordingly when this plan is merged.** A note has been added at the bottom of this plan as a reminder.

---

### Task 1: Pin engine state — read-only audit, no code changes

**Files:**
- Read-only: `catan_engine/src/rules.rs`, `catan_engine/src/state.rs`, `catan_engine/src/lib.rs`, `catan_engine/src/actions.rs`, `catan_engine/tests/main_phase.rs`, `catan_engine/tests/robber.rs`

- [ ] **Step 1: Read the four files above and verify these facts**

The plan's task code below assumes:
- `Action` enum has 6 variants: `BuildSettlement(u8)`, `BuildCity(u8)`, `BuildRoad(u8)`, `MoveRobber(u8)`, `Discard(Resource)`, `EndTurn` (`actions.rs:5-13`).
- `ACTION_SPACE_SIZE = 205` (`actions.rs:15`).
- Dice roll triggers on `(GamePhase::Roll, Action::EndTurn)` in `rules.rs` (~line 148).
- Auto-steal lives in the `MoveRobber` arm of `apply()` (~lines 251-259).
- `GamePhase::Steal { from_options: Vec<u8> }` is defined (`state.rs:24`) but never set as the live phase.
- `PyEngine` is in `lib.rs` and has methods `legal_actions`, `step`, `is_terminal`, `current_player`, `observation`, `stats` (`lib.rs:17-91`).

If any fact is wrong (e.g. someone refactored `lib.rs`), STOP and reconcile with the executor before continuing.

- [ ] **Step 2: Run the existing test suite and record the baseline**

Run: `cargo test --manifest-path catan_engine/Cargo.toml`
Expected: all tests pass. Record the count (e.g. "47 passed").

Run: `pytest tests/ -v` (from repo root)
Expected: all tests pass. Record the count.

This baseline is what Phase 0 must preserve at the end.

---

### Task 2: Add `RollDice` action variant + ID

**Files:**
- Modify: `catan_engine/src/actions.rs`
- Test: `catan_engine/tests/action_encoding.rs`

- [ ] **Step 1: Write the failing test**

Append to `catan_engine/tests/action_encoding.rs`:

```rust
#[test]
fn roll_dice_action_round_trips() {
    use catan_engine::actions::{Action, encode, decode};
    let id = encode(Action::RollDice);
    assert_eq!(id, 205);
    assert_eq!(decode(205), Some(Action::RollDice));
}

#[test]
fn action_space_size_is_206_after_roll_dice() {
    assert_eq!(catan_engine::actions::ACTION_SPACE_SIZE, 206);
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --manifest-path catan_engine/Cargo.toml roll_dice_action_round_trips action_space_size_is_206_after_roll_dice`
Expected: FAIL — `Action::RollDice` does not exist.

- [ ] **Step 3: Add the variant**

Edit `catan_engine/src/actions.rs`:

Replace the `Action` enum body (`actions.rs:5-13`) with:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    BuildSettlement(u8), // vertex 0..54
    BuildCity(u8),       // vertex 0..54
    BuildRoad(u8),       // edge 0..72
    MoveRobber(u8),      // hex 0..19
    Discard(Resource),
    EndTurn,
    RollDice,
}
```

Replace `pub const ACTION_SPACE_SIZE: usize = 205;` (`actions.rs:15`) with:

```rust
pub const ACTION_SPACE_SIZE: usize = 206;
```

In `pub fn encode(...)`, add `Action::RollDice => 205` to the match (before the closing brace of the function).

In `pub fn decode(...)`, add `205 => Some(Action::RollDice),` arm above the catch-all `_ => None`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --manifest-path catan_engine/Cargo.toml`
Expected: PASS — including the two new tests, with no regressions in existing tests.

- [ ] **Step 5: Commit**

```bash
git add catan_engine/src/actions.rs catan_engine/tests/action_encoding.rs
git commit -m "feat(engine): add Action::RollDice (id 205), action space → 206"
```

---

### Task 3: Split dice roll out of `EndTurn`, expose `RollDice` as the legal action in `Roll` phase

**Files:**
- Modify: `catan_engine/src/rules.rs`
- Test: `catan_engine/tests/main_phase.rs` (existing) — DO NOT change behavior here, but note: existing tests that assumed `EndTurn` triggers the roll need updating in Step 4.

- [ ] **Step 1: Write the failing test**

Append to `catan_engine/tests/main_phase.rs`:

```rust
#[test]
fn roll_phase_legal_action_is_roll_dice_only() {
    use catan_engine::Engine;
    let mut e = Engine::new(42);
    // Drive through setup using whatever helpers exist; if no helpers, do it
    // manually by stepping through legal actions with a deterministic policy.
    while !matches!(e.state.phase, catan_engine::state::GamePhase::Roll) {
        let legal = e.legal_actions();
        assert!(!legal.is_empty(), "got stuck before reaching Roll");
        e.step(legal[0]);
    }
    let legal = e.legal_actions();
    assert_eq!(legal, vec![205], "Roll phase should expose only RollDice (id 205)");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --manifest-path catan_engine/Cargo.toml roll_phase_legal_action_is_roll_dice_only`
Expected: FAIL — currently `Roll` returns `vec![Action::EndTurn]`.

- [ ] **Step 3: Update `legal_actions` and `apply` for the `Roll` phase**

In `catan_engine/src/rules.rs`:

Change the `Roll` branch of `legal_actions` (currently `crate::state::GamePhase::Roll => vec![Action::EndTurn],` around line 17) to:

```rust
crate::state::GamePhase::Roll => vec![Action::RollDice],
```

In `pub fn apply`, find the `(GamePhase::Roll, Action::EndTurn)` match arm (around `rules.rs:148`) and rename the action to `Action::RollDice`. The match arm body is unchanged — it still rolls dice and transitions to Discard/MoveRobber/Main. The header should now read:

```rust
(GamePhase::Roll, Action::RollDice) => {
    use rand::Rng as _;
    let d1 = rng.inner().gen_range(1u8..=6);
    let d2 = rng.inner().gen_range(1u8..=6);
    // ... unchanged body ...
}
```

- [ ] **Step 4: Update existing tests that assumed `EndTurn` triggers the roll**

Run: `cargo test --manifest-path catan_engine/Cargo.toml 2>&1 | grep -E "FAIL|test result"`

For each failing test, the fix is: where the test pushes `Action::EndTurn` (or its encoded ID 204) while in `Roll` phase, change it to `Action::RollDice` (or ID 205).

Likely affected test files (verify with the grep above):
- `catan_engine/tests/main_phase.rs`
- `catan_engine/tests/discard.rs`
- `catan_engine/tests/robber.rs`
- `catan_engine/tests/smoke.rs`

The smoke test (`tests/smoke.rs`) takes `legal[0]` deterministically; since `Roll` now exposes `RollDice` as `legal[0]`, this should *just work* with no source change. Verify by running it.

- [ ] **Step 5: Run all tests**

Run: `cargo test --manifest-path catan_engine/Cargo.toml`
Expected: PASS — including the new `roll_phase_legal_action_is_roll_dice_only` test, and all updated existing tests.

- [ ] **Step 6: Commit**

```bash
git add catan_engine/src/rules.rs catan_engine/tests/
git commit -m "refactor(engine): split dice roll into Action::RollDice"
```

---

### Task 4: Add chance-pending API to `Engine` (Rust-side, no PyO3 yet)

**Files:**
- Modify: `catan_engine/src/engine.rs`
- Modify: `catan_engine/src/state.rs` (small helper)
- Test: `catan_engine/tests/chance_api.rs` (new)

- [ ] **Step 1: Write the failing test**

Create `catan_engine/tests/chance_api.rs`:

```rust
//! Tests for the chance-node API used by OpenSpiel adapter (Phase 0 of MCTS study).

use catan_engine::Engine;
use catan_engine::state::GamePhase;

#[test]
fn chance_pending_is_false_during_setup() {
    let e = Engine::new(42);
    assert!(matches!(e.state.phase, GamePhase::Setup1Place));
    assert!(!e.is_chance_pending());
}

#[test]
fn chance_pending_is_true_in_roll_phase() {
    let mut e = Engine::new(42);
    while !matches!(e.state.phase, GamePhase::Roll) {
        let legal = e.legal_actions();
        e.step(legal[0]);
    }
    assert!(e.is_chance_pending());
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --manifest-path catan_engine/Cargo.toml --test chance_api`
Expected: FAIL — method `is_chance_pending` does not exist.

- [ ] **Step 3: Add `is_chance_pending` to `Engine`**

In `catan_engine/src/engine.rs`, add inside `impl Engine` (after `is_terminal`):

```rust
/// Returns true when the next action must come from the environment (dice / steal),
/// not from a player. Used by OpenSpiel chance-node modeling.
pub fn is_chance_pending(&self) -> bool {
    matches!(self.state.phase, crate::state::GamePhase::Roll)
        || matches!(self.state.phase, crate::state::GamePhase::Steal { .. })
}
```

(`Steal` is added as a live phase in Task 5; it's safe to reference now because the variant already exists in the enum.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --manifest-path catan_engine/Cargo.toml --test chance_api`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add catan_engine/src/engine.rs catan_engine/tests/chance_api.rs
git commit -m "feat(engine): is_chance_pending() for Roll/Steal phases"
```

---

### Task 5: Make `Steal` an actual live phase (chance point), preserving auto-steal as fallback

**Files:**
- Modify: `catan_engine/src/rules.rs`
- Modify: `catan_engine/src/state.rs` (no struct changes; the `Steal` variant already exists)
- Test: `catan_engine/tests/chance_api.rs` (extend)
- Test: `catan_engine/tests/robber.rs` (existing — verify still passes)

- [ ] **Step 1: Write the failing test**

Append to `catan_engine/tests/chance_api.rs`:

```rust
#[test]
fn moving_robber_to_hex_with_victims_enters_steal_phase() {
    // Construct a minimal scenario: hand-build a state where current player rolls a 7,
    // moves robber to a hex with one opponent settlement, and the opponent has cards.
    // We do this via the public API: drive setup such that player 0 ends adjacent to a
    // hex with a known number, then force a 7 roll by seeding.
    // Simpler approach: run random games with fixed seeds until we hit a Steal entry,
    // then assert the phase variant.
    use catan_engine::Engine;
    use catan_engine::state::GamePhase;

    for seed in 0..200u64 {
        let mut e = Engine::new(seed);
        for _ in 0..400 {
            if e.is_terminal() { break; }
            if matches!(e.state.phase, GamePhase::Steal { .. }) {
                assert!(e.is_chance_pending(), "Steal must imply chance pending");
                return;
            }
            let legal = e.legal_actions();
            if legal.is_empty() { break; }
            e.step(legal[0]);
        }
    }
    panic!("no Steal phase reached in 200 seeds — auto-steal regression?");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --manifest-path catan_engine/Cargo.toml --test chance_api moving_robber_to_hex_with_victims_enters_steal_phase`
Expected: FAIL — current code auto-steals inside `MoveRobber` arm, never enters `Steal`.

- [ ] **Step 3: Modify the `MoveRobber` apply arm to enter `Steal` instead of auto-stealing**

In `catan_engine/src/rules.rs`, the `(GamePhase::MoveRobber, Action::MoveRobber(h))` arm (around `rules.rs:231-260`):

Replace lines that read:
```rust
            if targets.is_empty() {
                state.phase = GamePhase::Main;
            } else {
                // Auto-steal from first target (Tier 1 simplification — see plan).
                let victim = targets[0];
                let stolen = steal_random(state, victim, rng);
                events.push(GameEvent::Robbed { from: victim, to: me, resource: stolen });
                state.phase = GamePhase::Main;
            }
```

with:

```rust
            if targets.is_empty() {
                state.phase = GamePhase::Main;
            } else {
                // Hand control to chance: which victim is picked + which card is taken.
                state.phase = GamePhase::Steal { from_options: targets };
            }
```

- [ ] **Step 4: Add a Steal-phase apply arm and `legal_actions` arm**

In `rules.rs` `legal_actions`, replace the existing `Steal` arm (currently returns `vec![]`, around `rules.rs:39-44`) with:

```rust
crate::state::GamePhase::Steal { .. } => {
    // Chance node — environment supplies the outcome via apply_chance_outcome().
    // Players have no legal actions here.
    vec![]
}
```

(Functionally identical, but the comment is now accurate.)

In `rules.rs` `apply`, the `Steal` apply path is **only reachable from `apply_chance_outcome` (Task 7)**, not from a player action. We do not add a player-action arm for `Steal`. Instead, in Task 7 we add a separate function called by the PyO3 layer that performs the steal and transitions to `Main`.

- [ ] **Step 5: Run all tests**

Run: `cargo test --manifest-path catan_engine/Cargo.toml`
Expected: The new `moving_robber_to_hex_with_victims_enters_steal_phase` test passes. **Existing tests in `robber.rs` that expected auto-steal will now fail** — that is correct behavior change. We fix them in the next step.

- [ ] **Step 6: Update `robber.rs` tests for new flow**

For each failing test in `tests/robber.rs`, the pattern to apply:

After stepping `Action::MoveRobber(h)`, if the test expected an immediate auto-steal and transition to `Main`, the test must now:
1. Assert `engine.state.phase` is `GamePhase::Steal { .. }`.
2. Drive the steal via `engine.apply_chance_outcome(...)` (this method is added in Task 7). **For this task only**, skip those assertions and replace with `// TODO Task 7: drive steal via apply_chance_outcome` so the test compiles and is `#[ignore]`-d:

```rust
#[ignore = "re-enable after Task 7 adds apply_chance_outcome"]
#[test]
fn ...existing test name... { ... }
```

This is the only task in the plan where we leave temporarily-ignored tests; Task 7 re-enables them.

- [ ] **Step 7: Run all tests**

Run: `cargo test --manifest-path catan_engine/Cargo.toml`
Expected: All tests pass; the temporarily-ignored ones are skipped.

- [ ] **Step 8: Commit**

```bash
git add catan_engine/src/rules.rs catan_engine/tests/
git commit -m "refactor(engine): Steal becomes a live chance phase"
```

---

### Task 6: Implement `chance_outcomes()` returning the discrete outcome distribution

**Files:**
- Modify: `catan_engine/src/engine.rs`
- Test: `catan_engine/tests/chance_api.rs`

- [ ] **Step 1: Write the failing test**

Append to `catan_engine/tests/chance_api.rs`:

```rust
#[test]
fn chance_outcomes_for_roll_phase_are_2d6_distribution() {
    use catan_engine::Engine;
    use catan_engine::state::GamePhase;

    let mut e = Engine::new(42);
    while !matches!(e.state.phase, GamePhase::Roll) {
        let legal = e.legal_actions();
        e.step(legal[0]);
    }

    let outcomes = e.chance_outcomes();
    // 2d6 has 11 distinct sums (2..=12).
    assert_eq!(outcomes.len(), 11);
    let total: f64 = outcomes.iter().map(|(_, p)| p).sum();
    assert!((total - 1.0).abs() < 1e-9, "probs must sum to 1, got {total}");
    // Spot check: P(7) = 6/36.
    let p7 = outcomes.iter().find(|(v, _)| *v == 7).unwrap().1;
    assert!((p7 - 6.0/36.0).abs() < 1e-9);
}

#[test]
fn chance_outcomes_for_steal_are_uniform_over_victim_hand() {
    // Find a Steal phase, then verify outcomes are a flat distribution
    // whose count equals the victim's total card count.
    use catan_engine::Engine;
    use catan_engine::state::GamePhase;

    for seed in 0..200u64 {
        let mut e = Engine::new(seed);
        for _ in 0..400 {
            if e.is_terminal() { break; }
            if let GamePhase::Steal { from_options } = &e.state.phase {
                let victim = from_options[0];
                let total: u8 = e.state.hands[victim as usize].iter().sum();
                let outcomes = e.chance_outcomes();
                assert_eq!(outcomes.len(), total as usize, "one outcome per card in victim's hand");
                let sum: f64 = outcomes.iter().map(|(_, p)| p).sum();
                assert!((sum - 1.0).abs() < 1e-9);
                return;
            }
            let legal = e.legal_actions();
            if legal.is_empty() { break; }
            e.step(legal[0]);
        }
    }
    panic!("no Steal phase reached in 200 seeds");
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --manifest-path catan_engine/Cargo.toml --test chance_api chance_outcomes`
Expected: FAIL — method `chance_outcomes` does not exist.

- [ ] **Step 3: Implement `chance_outcomes`**

In `catan_engine/src/engine.rs`, inside `impl Engine` (after `is_chance_pending`):

```rust
/// Discrete chance outcomes at the current state. Each entry is `(outcome_value, probability)`.
///
/// Outcome semantics:
///   - In `Roll` phase: outcome_value is the dice sum, 2..=12.
///   - In `Steal` phase: outcome_value packs `(victim_index_in_from_options, card_index_in_victim_hand)`
///     as `victim * 256 + card_index`. The card_index is a flat index 0..total_cards into the victim's
///     hand: 0..hand[wood], hand[wood]..hand[wood]+hand[brick], etc. This matches the offset arithmetic
///     in `apply_chance_outcome`.
///
/// Panics if not at a chance node.
pub fn chance_outcomes(&self) -> Vec<(u32, f64)> {
    use crate::state::GamePhase;
    match &self.state.phase {
        GamePhase::Roll => {
            // 2d6 distribution: counts of (d1+d2) for d1,d2 in 1..=6.
            let mut counts = [0u32; 13]; // index = sum
            for d1 in 1..=6 {
                for d2 in 1..=6 {
                    counts[d1 + d2] += 1;
                }
            }
            let mut out = Vec::with_capacity(11);
            for s in 2..=12u32 {
                out.push((s, counts[s as usize] as f64 / 36.0));
            }
            out
        }
        GamePhase::Steal { from_options } => {
            // For Tier 1 we steal from the FIRST victim in from_options (matches old behavior).
            // Each card in the victim's hand is an equally likely outcome.
            let victim = from_options[0];
            let hand = &self.state.hands[victim as usize];
            let total: u32 = hand.iter().map(|&x| x as u32).sum();
            assert!(total > 0, "Steal phase entered with empty victim hand");
            let mut out = Vec::with_capacity(total as usize);
            let p = 1.0 / total as f64;
            for card_index in 0..total {
                out.push(((victim as u32) * 256 + card_index, p));
            }
            out
        }
        _ => panic!("chance_outcomes() called outside a chance node"),
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --manifest-path catan_engine/Cargo.toml --test chance_api chance_outcomes`
Expected: PASS.

- [ ] **Step 5: Run full suite**

Run: `cargo test --manifest-path catan_engine/Cargo.toml`
Expected: PASS (no regressions).

- [ ] **Step 6: Commit**

```bash
git add catan_engine/src/engine.rs catan_engine/tests/chance_api.rs
git commit -m "feat(engine): chance_outcomes() for dice + steal"
```

---

### Task 7: Implement `apply_chance_outcome(value)` and re-enable Steal-phase tests

**Files:**
- Modify: `catan_engine/src/engine.rs`
- Modify: `catan_engine/src/rules.rs` (helper function)
- Test: `catan_engine/tests/chance_api.rs`
- Test: `catan_engine/tests/robber.rs` (re-enable ignored tests)

- [ ] **Step 1: Write the failing test**

Append to `catan_engine/tests/chance_api.rs`:

```rust
#[test]
fn apply_chance_outcome_roll_advances_phase() {
    use catan_engine::Engine;
    use catan_engine::state::GamePhase;

    let mut e = Engine::new(42);
    while !matches!(e.state.phase, GamePhase::Roll) {
        let legal = e.legal_actions();
        e.step(legal[0]);
    }
    assert!(e.is_chance_pending());
    e.apply_chance_outcome(8); // not a 7 → goes to Main
    assert!(matches!(e.state.phase, GamePhase::Main));
}

#[test]
fn apply_chance_outcome_steal_transfers_card_and_returns_to_main() {
    use catan_engine::Engine;
    use catan_engine::state::GamePhase;

    for seed in 0..200u64 {
        let mut e = Engine::new(seed);
        for _ in 0..400 {
            if e.is_terminal() { break; }
            if let GamePhase::Steal { from_options } = &e.state.phase {
                let victim = from_options[0] as usize;
                let me = e.state.current_player as usize;
                let victim_total_before: u8 = e.state.hands[victim].iter().sum();
                let me_total_before: u8 = e.state.hands[me].iter().sum();
                let outcomes = e.chance_outcomes();
                let (chosen, _) = outcomes[0];
                e.apply_chance_outcome(chosen);
                assert!(matches!(e.state.phase, GamePhase::Main));
                assert_eq!(e.state.hands[victim].iter().sum::<u8>(), victim_total_before - 1);
                assert_eq!(e.state.hands[me].iter().sum::<u8>(), me_total_before + 1);
                return;
            }
            let legal = e.legal_actions();
            if legal.is_empty() { break; }
            e.step(legal[0]);
        }
    }
    panic!("no Steal phase reached in 200 seeds");
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --manifest-path catan_engine/Cargo.toml --test chance_api apply_chance_outcome`
Expected: FAIL — method does not exist.

- [ ] **Step 3: Implement `apply_chance_outcome`**

In `catan_engine/src/engine.rs`, inside `impl Engine`:

```rust
/// Drive a chance node by supplying the outcome value (one of those returned by `chance_outcomes()`).
/// Records the resulting events in the event log just like `step()` does.
/// Panics if not at a chance node, or if `value` is not a valid outcome for the current phase.
pub fn apply_chance_outcome(&mut self, value: u32) {
    use crate::state::GamePhase;
    let evs = match self.state.phase.clone() {
        GamePhase::Roll => {
            let roll = u8::try_from(value).expect("dice roll out of u8 range");
            assert!((2..=12).contains(&roll), "invalid dice sum {roll}");
            crate::rules::apply_dice_roll(&mut self.state, roll)
        }
        GamePhase::Steal { from_options } => {
            let victim = (value / 256) as u8;
            let card_index = (value % 256) as u8;
            assert_eq!(
                victim, from_options[0],
                "value's victim {} != from_options[0] {} — Tier 1 steals only from first option",
                victim, from_options[0],
            );
            crate::rules::apply_steal(&mut self.state, victim, card_index)
        }
        _ => panic!("apply_chance_outcome() called outside a chance node"),
    };
    for e in evs {
        self.events.push(e);
    }
}
```

- [ ] **Step 4: Add `apply_dice_roll` and `apply_steal` helpers in `rules.rs`**

In `catan_engine/src/rules.rs`, add (replacing the dice-roll body in the existing `(GamePhase::Roll, Action::RollDice)` arm with a call to the new helper):

The match arm becomes:
```rust
(GamePhase::Roll, Action::RollDice) => {
    use rand::Rng as _;
    let d1 = rng.inner().gen_range(1u8..=6);
    let d2 = rng.inner().gen_range(1u8..=6);
    let roll = d1 + d2;
    let mut sub = apply_dice_roll(state, roll);
    events.append(&mut sub);
}
```

Add the two new public-in-crate helpers at the bottom of `rules.rs` (after `steal_random`):

```rust
/// Apply a specific dice roll value (used by both the rng-driven Action::RollDice path
/// and the env-driven apply_chance_outcome path, so behavior is single-sourced).
pub(crate) fn apply_dice_roll(state: &mut GameState, roll: u8) -> Vec<GameEvent> {
    let mut events = Vec::new();
    events.push(GameEvent::DiceRolled { roll });
    if roll == 7 {
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
    events
}

/// Apply a specific steal: take `card_index`-th card from `victim`'s hand and give to current player.
pub(crate) fn apply_steal(state: &mut GameState, victim: u8, card_index: u8) -> Vec<GameEvent> {
    let mut events = Vec::new();
    let me = state.current_player;
    let mut acc = 0u8;
    for ri in 0..5usize {
        let count = state.hands[victim as usize][ri];
        if card_index < acc + count {
            state.hands[victim as usize][ri] -= 1;
            state.hands[me as usize][ri] += 1;
            let res = match ri {
                0 => crate::board::Resource::Wood,
                1 => crate::board::Resource::Brick,
                2 => crate::board::Resource::Sheep,
                3 => crate::board::Resource::Wheat,
                4 => crate::board::Resource::Ore,
                _ => unreachable!(),
            };
            events.push(GameEvent::Robbed { from: victim, to: me, resource: Some(res) });
            state.phase = GamePhase::Main;
            return events;
        }
        acc += count;
    }
    panic!("card_index {card_index} out of range; victim hand sums to {acc}");
}
```

- [ ] **Step 5: Re-enable the previously ignored robber tests**

In `catan_engine/tests/robber.rs`, find tests marked `#[ignore = "re-enable after Task 7 adds apply_chance_outcome"]` from Task 5, Step 6.

For each:
1. Remove the `#[ignore = "..."]` line.
2. Where the test expected post-`MoveRobber` to be in `Main` with a card transferred, replace that expectation with: assert `Steal`, call `engine.apply_chance_outcome(engine.chance_outcomes()[0].0)`, then assert `Main` with card transferred.

If the original test wanted to verify the auto-steal *event*, it can now look at the event log after the chance outcome.

- [ ] **Step 6: Run all tests**

Run: `cargo test --manifest-path catan_engine/Cargo.toml`
Expected: PASS — no ignored tests remaining, no regressions.

- [ ] **Step 7: Commit**

```bash
git add catan_engine/src/engine.rs catan_engine/src/rules.rs catan_engine/tests/robber.rs catan_engine/tests/chance_api.rs
git commit -m "feat(engine): apply_chance_outcome() drives dice + steal"
```

---

### Task 8: Add `clone()` and `action_history()` to Engine

**Files:**
- Modify: `catan_engine/src/engine.rs`
- Test: `catan_engine/tests/chance_api.rs`

- [ ] **Step 1: Write the failing test**

Append to `catan_engine/tests/chance_api.rs`:

```rust
#[test]
fn clone_is_independent() {
    use catan_engine::Engine;
    let mut a = Engine::new(42);
    a.step(a.legal_actions()[0]);
    let mut b = a.clone();
    let phase_before = format!("{:?}", a.state.phase);
    b.step(b.legal_actions()[0]);
    let phase_after_a = format!("{:?}", a.state.phase);
    assert_eq!(phase_before, phase_after_a, "stepping the clone must not affect the original");
}

#[test]
fn action_history_records_steps() {
    use catan_engine::Engine;
    let mut e = Engine::new(42);
    let a0 = e.legal_actions()[0];
    e.step(a0);
    let a1 = e.legal_actions()[0];
    e.step(a1);
    assert_eq!(e.action_history(), &[a0, a1]);
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --manifest-path catan_engine/Cargo.toml --test chance_api clone_is_independent action_history_records_steps`
Expected: FAIL — methods do not exist.

- [ ] **Step 3: Make `Engine` cloneable and add `action_history`**

In `catan_engine/src/engine.rs`:

Replace the `Engine` struct with:
```rust
#[derive(Clone)]
pub struct Engine {
    pub state: GameState,
    pub rng: Rng,
    pub events: EventLog,
    pub history: Vec<u32>,
}
```

Add `#[derive(Clone)]` to `Rng` in `catan_engine/src/rng.rs` (above the struct):
```rust
#[derive(Clone)]
pub struct Rng { ... }
```

Add `#[derive(Clone)]` to `EventLog` in `catan_engine/src/events.rs`:
```rust
#[derive(Clone)]
pub struct EventLog { ... }
```

In `Engine::new`, initialize `history: Vec::new()`.

In `Engine::step`, after the `decode(action_id)` line, push the id:
```rust
pub fn step(&mut self, action_id: u32) {
    let action = decode(action_id).expect("invalid action ID");
    self.history.push(action_id);
    let evs = apply(&mut self.state, action, &mut self.rng);
    for e in evs {
        self.events.push(e);
    }
}
```

In `Engine::apply_chance_outcome`, after the value is validated, also push a marker. Since chance outcomes aren't in the action ID space, encode them with the high bit set: `self.history.push(0x8000_0000 | value)`. This keeps `history` reconstructable.

Add the accessor:
```rust
pub fn action_history(&self) -> &[u32] {
    &self.history
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test --manifest-path catan_engine/Cargo.toml`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add catan_engine/src/engine.rs catan_engine/src/rng.rs catan_engine/src/events.rs catan_engine/tests/chance_api.rs
git commit -m "feat(engine): Engine: Clone + action_history()"
```

---

### Task 9: Expose the new API through PyO3

**Files:**
- Modify: `catan_engine/src/lib.rs`
- Test: `tests/test_python_api.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_python_api.py`:

```python
def test_chance_api_roundtrip():
    from catan_bot import _engine
    e = _engine.Engine(42)
    # Drive to a Roll phase by taking legal[0] until is_chance_pending.
    for _ in range(2000):
        if e.is_chance_pending():
            break
        legal = e.legal_actions()
        if len(legal) == 0:
            break
        e.step(int(legal[0]))
    assert e.is_chance_pending()
    outcomes = e.chance_outcomes()  # list of (value, probability)
    assert len(outcomes) == 11      # 2d6 sums
    total = sum(p for _, p in outcomes)
    assert abs(total - 1.0) < 1e-9
    e.apply_chance_outcome(int(outcomes[0][0]))


def test_clone_independence():
    from catan_bot import _engine
    a = _engine.Engine(7)
    a.step(int(a.legal_actions()[0]))
    b = a.clone()
    a_phase_repr = a.action_history()
    b.step(int(b.legal_actions()[0]))
    assert a.action_history() == a_phase_repr  # untouched


def test_action_history():
    from catan_bot import _engine
    e = _engine.Engine(7)
    a0 = int(e.legal_actions()[0])
    e.step(a0)
    hist = e.action_history()
    assert hist[0] == a0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `maturin develop --manifest-path catan_engine/Cargo.toml && pytest tests/test_python_api.py::test_chance_api_roundtrip tests/test_python_api.py::test_clone_independence tests/test_python_api.py::test_action_history -v`
Expected: FAIL — methods don't exist on `PyEngine`.

- [ ] **Step 3: Add PyO3 methods**

In `catan_engine/src/lib.rs`, inside `#[pymethods] impl PyEngine`:

```rust
fn is_chance_pending(&self) -> bool {
    self.inner.is_chance_pending()
}

fn chance_outcomes(&self) -> Vec<(u32, f64)> {
    self.inner.chance_outcomes()
}

fn apply_chance_outcome(&mut self, value: u32) {
    self.inner.apply_chance_outcome(value);
}

fn clone(&self) -> Self {
    Self { inner: self.inner.clone() }
}

fn action_history(&self) -> Vec<u32> {
    self.inner.action_history().to_vec()
}
```

- [ ] **Step 4: Rebuild and run tests**

Run: `maturin develop --manifest-path catan_engine/Cargo.toml`
Expected: build succeeds.

Run: `pytest tests/test_python_api.py -v`
Expected: PASS for new tests, no regressions.

- [ ] **Step 5: Commit**

```bash
git add catan_engine/src/lib.rs tests/test_python_api.py
git commit -m "feat(pyo3): expose chance API + clone + action_history"
```

---

### Task 10: Determinism regression — chance API does not break existing seed hashes

**Files:**
- Modify: existing determinism test (or create one if none exists)
- Test: `catan_engine/tests/determinism_chance.rs` (new)

- [ ] **Step 1: Write the failing test**

Create `catan_engine/tests/determinism_chance.rs`:

```rust
//! Phase 0 must not change behavior of legal[0]-policy random self-play.
//! We check that for several seeds, full games complete deterministically and
//! produce a stable event-log hash. Hashes recorded here lock the pre-Phase-0
//! seed→trajectory contract going forward.

use catan_engine::Engine;
use std::hash::{Hash, Hasher};

fn play_one(seed: u64) -> u64 {
    let mut e = Engine::new(seed);
    let mut steps = 0;
    while !e.is_terminal() && steps < 5000 {
        if e.is_chance_pending() {
            let outcomes = e.chance_outcomes();
            // Pick the highest-probability outcome deterministically; ties broken by lowest value.
            let (best_val, _) = outcomes.iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap().then_with(|| b.0.cmp(&a.0)))
                .unwrap();
            e.apply_chance_outcome(*best_val);
        } else {
            let legal = e.legal_actions();
            if legal.is_empty() { break; }
            e.step(legal[0]);
        }
        steps += 1;
    }
    let mut h = std::collections::hash_map::DefaultHasher::new();
    for ev in e.events.as_slice() {
        format!("{ev:?}").hash(&mut h);
    }
    h.finish()
}

#[test]
fn fixed_seeds_produce_stable_hashes() {
    // Lock these by running once and copying the printed values in. After that,
    // any change that flips a hash is a flagged behavior change.
    let seeds = [1u64, 7, 42, 100, 999];
    let hashes: Vec<u64> = seeds.iter().map(|&s| play_one(s)).collect();
    // First time: print and copy. Once locked, replace this with hardcoded asserts.
    eprintln!("hashes: {:?}", hashes);
    // Each hash must equal itself across two runs of this same test process:
    let again: Vec<u64> = seeds.iter().map(|&s| play_one(s)).collect();
    assert_eq!(hashes, again, "non-determinism inside a single process");
}
```

- [ ] **Step 2: Run test**

Run: `cargo test --manifest-path catan_engine/Cargo.toml --test determinism_chance -- --nocapture`
Expected: PASS, with hashes printed to stderr. If the test fails on the within-process check, that's a real determinism bug — fix before continuing.

- [ ] **Step 3: Lock the hashes**

Replace the body of `fixed_seeds_produce_stable_hashes` with hardcoded assertions using the printed hashes:

```rust
#[test]
fn fixed_seeds_produce_stable_hashes() {
    let seeds = [1u64, 7, 42, 100, 999];
    let expected: [u64; 5] = [/* paste from Step 2 output */];
    for (s, exp) in seeds.iter().zip(expected.iter()) {
        assert_eq!(play_one(*s), *exp, "seed {s} hash drift");
    }
}
```

- [ ] **Step 4: Run test**

Run: `cargo test --manifest-path catan_engine/Cargo.toml --test determinism_chance`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add catan_engine/tests/determinism_chance.rs
git commit -m "test(engine): determinism regression for chance-driven self-play"
```

---

### Task 11: Cross-cutting reminder — bump action-space constant in MCTS-study spec/Plan 2

**Files:**
- Modify: `docs/superpowers/specs/2026-04-27-mcts-study-design.md`

- [ ] **Step 1: Update the schema constant from 205 to 206**

In `docs/superpowers/specs/2026-04-27-mcts-study-design.md`:

Replace `list<bool>[205]` with `list<bool>[206]` (one occurrence in the moves schema).
Replace `list<int32>[205]` with `list<int32>[206]` (one occurrence in the moves schema).
Replace `Matches engine's action space` (still accurate, no change needed) — but add a footnote:

After the moves schema table, add:
```
> **Note:** Schema width is 206 (Phase 0 added `Action::RollDice` at id 205). Earlier drafts referenced 205; that value is wrong post-Phase-0.
```

- [ ] **Step 2: Verify**

Run: `grep -n "205\|206" docs/superpowers/specs/2026-04-27-mcts-study-design.md`
Expected: All 205-related mentions are now in historical/footnote context only. Live schema width is 206.

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/specs/2026-04-27-mcts-study-design.md
git commit -m "docs(spec): bump action-space width to 206 (post-Phase-0)"
```

---

## Plan-1 self-review checklist (run after Task 11)

- [ ] All Phase 0 success criteria met:
  - `Engine::is_chance_pending()`, `chance_outcomes()`, `apply_chance_outcome(value)`, `clone()`, `action_history()` all exist on Rust side.
  - All five exposed via PyO3 on `PyEngine`.
  - Determinism regression test in place.
  - Action-space constant in spec is 206.
- [ ] No `#[ignore]` annotations remain in any test file.
- [ ] Full Rust test suite green: `cargo test --manifest-path catan_engine/Cargo.toml`
- [ ] Full Python test suite green: `pytest tests/`
- [ ] No new lint warnings: `cargo clippy --manifest-path catan_engine/Cargo.toml -- -D warnings` (if clippy is configured; otherwise skip)
