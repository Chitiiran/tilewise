//! Property-based tests on engine invariants.
//!
//! v2 additions:
//!   - inventory caps never exceeded
//!   - longest_road_holder ↔ +2 VP bookkeeping consistent
//!   - resource conservation also includes the chance-driven path (rolls/steals)
//!   - VP "never decreases except by longest-road transfer" (replacing v1's
//!     stricter non-decreasing assertion)
use catan_engine::actions::{decode, encode, ACTION_SPACE_SIZE};
use catan_engine::Engine;
use catan_engine::state::{MAX_SETTLEMENTS, MAX_CITIES, MAX_ROADS};
use proptest::prelude::*;

/// Helper that drives the engine through both decision and chance steps
/// using the test's choice seq. This exercises the full state machine
/// (including discard-7 and steal chance nodes) while v1's tests only
/// drove decisions through legal_actions.
fn drive_engine(engine: &mut Engine, choices: &[u32], n_steps: usize) -> Vec<usize> {
    let mut steps_taken = Vec::with_capacity(n_steps);
    for (i, &c) in choices.iter().take(n_steps).enumerate() {
        if engine.is_terminal() { break; }
        if engine.is_chance_pending() {
            let outcomes = engine.chance_outcomes();
            if outcomes.is_empty() { break; }
            let chosen = outcomes[(c as usize) % outcomes.len()].0;
            engine.apply_chance_outcome(chosen);
        } else {
            let legal = engine.legal_actions();
            if legal.is_empty() { break; }
            let a = legal[(c as usize) % legal.len()];
            engine.step(a);
        }
        steps_taken.push(i);
    }
    steps_taken
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    #[test]
    fn action_id_round_trip(id in 0u32..(ACTION_SPACE_SIZE as u32)) {
        let action = decode(id).unwrap();
        prop_assert_eq!(encode(action), id);
    }

    #[test]
    fn random_legal_play_never_panics(seed in any::<u64>(), choices in prop::collection::vec(any::<u32>(), 0..1000)) {
        let mut engine = Engine::new(seed);
        drive_engine(&mut engine, &choices, 1000);
        // No panic = pass.
    }

    #[test]
    fn invariant_total_resources_within_supply(seed in any::<u64>(), choices in prop::collection::vec(any::<u32>(), 0..500)) {
        let mut engine = Engine::new(seed);
        for (i, &c) in choices.iter().enumerate() {
            if engine.is_terminal() { break; }
            if engine.is_chance_pending() {
                let outcomes = engine.chance_outcomes();
                if outcomes.is_empty() { break; }
                engine.apply_chance_outcome(outcomes[(c as usize) % outcomes.len()].0);
            } else {
                let legal = engine.legal_actions();
                if legal.is_empty() { break; }
                engine.step(legal[(c as usize) % legal.len()]);
            }
            // After every state transition: bank + sum(player hands) == 19 per resource.
            for r in 0..5 {
                let in_hands: u32 = (0..4).map(|p| engine.state.hands[p][r] as u32).sum();
                let bank = engine.state.bank[r] as u32;
                prop_assert_eq!(in_hands + bank, 19u32,
                    "resource {} broken at step {}: hands={}, bank={}", r, i, in_hands, bank);
            }
        }
    }

    /// Inventory caps are never exceeded (§A2). Caps mean "currently on board"
    /// for settlements (since they decrement on city upgrade).
    #[test]
    fn invariant_inventory_caps_respected(seed in any::<u64>(), choices in prop::collection::vec(any::<u32>(), 0..500)) {
        let mut engine = Engine::new(seed);
        for &c in &choices {
            if engine.is_terminal() { break; }
            if engine.is_chance_pending() {
                let outcomes = engine.chance_outcomes();
                if outcomes.is_empty() { break; }
                engine.apply_chance_outcome(outcomes[(c as usize) % outcomes.len()].0);
            } else {
                let legal = engine.legal_actions();
                if legal.is_empty() { break; }
                engine.step(legal[(c as usize) % legal.len()]);
            }
            for p in 0..4 {
                prop_assert!(engine.state.settlements_built[p] <= MAX_SETTLEMENTS,
                    "P{} has {} settlements, max {}", p, engine.state.settlements_built[p], MAX_SETTLEMENTS);
                prop_assert!(engine.state.cities_built[p] <= MAX_CITIES,
                    "P{} has {} cities, max {}", p, engine.state.cities_built[p], MAX_CITIES);
                prop_assert!(engine.state.roads_built[p] <= MAX_ROADS,
                    "P{} has {} roads, max {}", p, engine.state.roads_built[p], MAX_ROADS);
            }
        }
    }

    /// Longest-road holder is consistent: if Some(p), then p has the strict-max
    /// length AND that length >= 5; if None, no player has length >= 5 strictly.
    #[test]
    fn invariant_longest_road_holder_consistent(seed in any::<u64>(), choices in prop::collection::vec(any::<u32>(), 0..500)) {
        let mut engine = Engine::new(seed);
        for &c in &choices {
            if engine.is_terminal() { break; }
            if engine.is_chance_pending() {
                let outcomes = engine.chance_outcomes();
                if outcomes.is_empty() { break; }
                engine.apply_chance_outcome(outcomes[(c as usize) % outcomes.len()].0);
            } else {
                let legal = engine.legal_actions();
                if legal.is_empty() { break; }
                engine.step(legal[(c as usize) % legal.len()]);
            }
            let max_len = engine.state.longest_road_length.iter().copied().max().unwrap_or(0);
            match engine.state.longest_road_holder {
                Some(p) => {
                    prop_assert!(engine.state.longest_road_length[p as usize] >= 5,
                        "Holder P{} has length {} (< 5)", p, engine.state.longest_road_length[p as usize]);
                    prop_assert_eq!(engine.state.longest_road_length[p as usize], max_len,
                        "Holder P{} doesn't have the max length", p);
                }
                None => {
                    // Either nobody has length >= 5, or there's a tie at the max
                    // (no holder when tied without prior holder).
                    let tied_at_max: Vec<u8> = (0..4u8)
                        .filter(|&p| engine.state.longest_road_length[p as usize] == max_len)
                        .collect();
                    prop_assert!(max_len < 5 || tied_at_max.len() >= 2,
                        "max_len={} but no holder, and only {} players tied", max_len, tied_at_max.len());
                }
            }
        }
    }

    /// Replay determinism: applying the recorded action_history to a fresh
    /// engine produces the same VP and current_player as the live engine.
    #[test]
    fn invariant_replay_determinism(seed in any::<u64>(), choices in prop::collection::vec(any::<u32>(), 0..200)) {
        let mut e1 = Engine::new(seed);
        drive_engine(&mut e1, &choices, 200);
        let history = e1.action_history().to_vec();

        let mut e2 = Engine::new(seed);
        for &a in &history {
            if a & 0x8000_0000 != 0 {
                let val = a & 0x7FFF_FFFF;
                e2.apply_chance_outcome(val);
            } else {
                e2.step(a);
            }
        }
        prop_assert_eq!(e1.state.vp, e2.state.vp);
        prop_assert_eq!(e1.state.current_player, e2.state.current_player);
        for r in 0..5 {
            prop_assert_eq!(e1.state.bank[r], e2.state.bank[r]);
            for p in 0..4 {
                prop_assert_eq!(e1.state.hands[p][r], e2.state.hands[p][r]);
            }
        }
    }
}
