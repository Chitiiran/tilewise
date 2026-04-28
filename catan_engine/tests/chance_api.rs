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
