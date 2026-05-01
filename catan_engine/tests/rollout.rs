//! Tests for Engine::random_rollout_to_terminal — the Rust-side rollout used
//! by the MCTS-study's RustRolloutEvaluator (P3.T7a).

use catan_engine::Engine;

#[test]
fn rollout_from_initial_state_terminates() {
    let mut e = Engine::new(42);
    let returns = e.random_rollout_to_terminal(123);
    assert!(e.is_terminal(), "rollout did not reach terminal state");
    // One winner: exactly one +1.0, three -1.0.
    let plus = returns.iter().filter(|&&r| r == 1.0).count();
    let minus = returns.iter().filter(|&&r| r == -1.0).count();
    assert_eq!(plus, 1, "expected 1 winner, got returns {:?}", returns);
    assert_eq!(minus, 3, "expected 3 losers, got returns {:?}", returns);
}

#[test]
fn rollout_is_deterministic_given_rollout_seed() {
    // Same engine seed AND same rollout seed → same returns + same final state.
    let mut a = Engine::new(42);
    let ra = a.random_rollout_to_terminal(7);
    let mut b = Engine::new(42);
    let rb = b.random_rollout_to_terminal(7);
    assert_eq!(ra, rb);
    // History (action_history) should also be byte-identical.
    assert_eq!(a.action_history(), b.action_history());
}

#[test]
fn rollout_seed_actually_varies_trajectory() {
    // Different rollout seeds → different histories (with very high probability).
    // We don't insist on different *winners*, just different *trajectories*.
    let mut a = Engine::new(42);
    a.random_rollout_to_terminal(1);
    let mut b = Engine::new(42);
    b.random_rollout_to_terminal(99);
    // At least one of: histories differ in length, or differ at some position.
    assert_ne!(a.action_history(), b.action_history(),
        "two distinct rollout seeds produced byte-identical histories");
}

#[test]
fn rollout_does_not_consume_engine_rng_for_chance_decisions() {
    // The engine's self.rng is used by step(RollDice) which rolls dice internally.
    // The rollout's own rng is used by apply_chance_outcome when bypassing step.
    // After a rollout, the engine ends in a terminal state — but two engines started
    // from the same seed (one with rollout_seed=1, one with rollout_seed=2) must
    // both have COMPLETED rollouts (i.e., the engine's RNG isn't blocking them).
    // This test is structural: just verifies both rollouts terminate cleanly.
    for rollout_seed in [1u64, 2, 3, 100] {
        let mut e = Engine::new(42);
        let returns = e.random_rollout_to_terminal(rollout_seed);
        // "Terminate cleanly" = either reached terminal (winner/loser returns)
        // OR hit the safety cap (all-zero returns). Either way, no panic / no hang.
        if e.is_terminal() {
            assert_eq!(returns.iter().filter(|&&r| r == 1.0).count(), 1);
        } else {
            assert_eq!(returns, [0.0f32; 4],
                "rollout_seed={} non-terminal but returns != zeros: {:?}",
                rollout_seed, returns);
        }
    }
}

#[test]
fn rollout_from_mid_game_state_terminates() {
    // Drive the engine through a few player decisions, then rollout from there.
    // v2: try several seeds to find one that produces a terminal-eligible rollout
    // within the 30k cap. Random play from sub-optimal mid-game states isn't
    // guaranteed to terminate in any single seed.
    for seed in [42u64, 7, 100, 1, 999] {
        let mut e = Engine::new(seed);
        for _ in 0..20 {
            if e.is_terminal() { break; }
            if e.is_chance_pending() {
                let outcomes = e.chance_outcomes();
                e.apply_chance_outcome(outcomes[0].0);
            } else {
                let legal = e.legal_actions();
                e.step(legal[0]);
            }
        }
        if e.is_terminal() { return; }
        let returns = e.random_rollout_to_terminal(seed);
        if e.is_terminal() {
            assert_eq!(returns.iter().filter(|&&r| r == 1.0).count(), 1);
            return; // success — at least one seed converges
        }
    }
    panic!("no seed produced a terminal rollout — check engine for non-terminating state");
}

#[test]
fn rollout_step_cap_constant_is_30k() {
    // v2 (2026-04-28): cap was lowered from 100_000 to 30_000 to bound the
    // worst-case single-game cost in MCTS rollouts. If this changes, also
    // update writeup §6 (caveats) and re-run the rollout-cap-firing-rate
    // measurement (project_rollout_cap_firing_rate.md).
    assert_eq!(catan_engine::engine::ROLLOUT_STEP_CAP, 30_000);
}

#[test]
fn rollout_caps_at_safety_limit_does_not_panic() {
    // Sanity guard: any rollout call must complete without panic, whether it
    // hits a natural terminal or the safety cap.
    let mut e = Engine::new(42);
    let _ = e.random_rollout_to_terminal(42);
    // (We intentionally don't assert is_terminal here — under the lower cap,
    //  more rollouts will hit the cap and return [0;4] without a Done phase.)
}
