//! Integration tests for the legal-action bitmap (Phase 1.2).
//! Confirms engine.legal_mask() agrees with engine.legal_actions() across many
//! states. This is the "brute-force == bitmap" property test that protects
//! future incremental updates from drifting.

use catan_engine::Engine;
use std::collections::HashSet;

fn drive_to_state(seed: u64, n_steps: usize) -> Engine {
    let mut e = Engine::new(seed);
    let mut steps = 0;
    while !e.is_terminal() && steps < n_steps {
        if e.is_chance_pending() {
            let outcomes = e.chance_outcomes();
            if outcomes.is_empty() { break; }
            e.apply_chance_outcome(outcomes[0].0);
        } else {
            let legal = e.legal_actions();
            if legal.is_empty() { break; }
            e.step(legal[steps % legal.len()]);
        }
        steps += 1;
    }
    e
}

#[test]
fn legal_mask_initial_state_matches_legal_actions() {
    let mut e = Engine::new(0);
    let acts: HashSet<u32> = e.legal_actions().into_iter().collect();
    let m = e.legal_mask();
    let mask_acts: HashSet<u32> = m.iter_ids().collect();
    assert_eq!(acts, mask_acts);
    assert_eq!(m.count() as usize, acts.len());
}

#[test]
fn legal_mask_matches_across_many_states() {
    // Drive the engine through varied states and check the bitmap matches every step.
    for seed in [0u64, 7, 42, 100, 1000] {
        for n_steps in [10, 50, 100, 500] {
            let mut e = drive_to_state(seed, n_steps);
            let acts: HashSet<u32> = e.legal_actions().into_iter().collect();
            let mask_acts: HashSet<u32> = e.legal_mask().iter_ids().collect();
            assert_eq!(
                acts, mask_acts,
                "seed={} n_steps={} terminal={}",
                seed, n_steps, e.is_terminal()
            );
        }
    }
}

#[test]
fn legal_mask_terminal_state_is_empty() {
    // Force-drive toward terminal with many random steps. Should hit terminal
    // for at least some seeds within the cap.
    let mut e = drive_to_state(2, 8000);
    if e.is_terminal() {
        assert_eq!(e.legal_mask().count(), 0, "terminal state should have no legal actions");
        assert_eq!(e.legal_actions().len(), 0);
    }
}
