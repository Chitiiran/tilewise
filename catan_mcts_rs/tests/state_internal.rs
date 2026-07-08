//! Rust-internal property tests for the state layer (the second verification
//! class alongside the Python differential test in test_rust_engine_parity.py).

use catan_mcts_rs::rng::NpRng;
use catan_mcts_rs::state;
use catan_engine::Engine;

/// chance_outcomes probabilities sum to ~1.0 at every chance node encountered
/// in a random playout, and sample_chance always returns a listed outcome.
#[test]
fn chance_outcomes_sum_to_one_and_sample_is_valid() {
    for seed in 0..30u64 {
        let mut e = Engine::new(seed);
        let mut rng = NpRng::from_seed(seed);
        let mut steps = 0;
        while !e.is_terminal() && steps < 400 {
            if e.is_chance_pending() {
                let outs = e.chance_outcomes();
                let s: f64 = outs.iter().map(|&(_, p)| p).sum();
                assert!((s - 1.0).abs() < 1e-9, "seed {seed}: chance probs sum {s}");
                let chosen = state::sample_chance(&e, &mut rng);
                assert!(outs.iter().any(|&(v, _)| v == chosen));
                e.apply_chance_outcome(chosen);
            } else {
                let legal = e.legal_actions();
                if legal.is_empty() {
                    break;
                }
                e.step(legal[0]);
            }
            steps += 1;
        }
    }
}

/// Clone is independent: mutating a clone does not affect the original's
/// legal_actions / current_player / history.
#[test]
fn clone_is_independent() {
    let mut e = Engine::new(42);
    // advance past the opening chance/forced moves
    for _ in 0..10 {
        if e.is_terminal() {
            break;
        }
        if e.is_chance_pending() {
            let v = e.chance_outcomes()[0].0;
            e.apply_chance_outcome(v);
        } else {
            let la = e.legal_actions();
            e.step(la[0]);
        }
    }
    let before_cp = state::current_player(&e);
    let before_hist = e.action_history().to_vec();
    let mut c = e.clone();
    // mutate the clone (step cap: greedy la[0] play may not terminate — this
    // test only needs the clone mutated, not finished).
    let mut steps = 0;
    while !c.is_terminal() && steps < 2000 {
        if c.is_chance_pending() {
            let v = c.chance_outcomes()[0].0;
            c.apply_chance_outcome(v);
        } else {
            let la = c.legal_actions();
            if la.is_empty() {
                break;
            }
            c.step(la[0]);
        }
        steps += 1;
    }
    assert_eq!(state::current_player(&e), before_cp, "original cp changed");
    assert_eq!(e.action_history().to_vec(), before_hist, "original history changed");
}

/// returns_abs: non-terminal -> all zeros.
#[test]
fn returns_zero_when_not_terminal() {
    let e = Engine::new(7);
    assert_eq!(state::returns_abs(&e), [0.0; 4]);
}
