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

#[test]
fn moving_robber_to_hex_with_victims_enters_steal_phase() {
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
