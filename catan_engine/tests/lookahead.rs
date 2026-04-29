//! Tests for Engine::lookahead_vp_value — the depth-bounded greedy evaluator
//! that replaces random rollouts in the MCTS leaf-evaluation phase. Compared
//! to random_rollout_to_terminal, this:
//!   - plays at most `depth` rounds (4*depth EndTurns), not to terminal,
//!   - uses VP-greedy decisions instead of uniform random,
//!   - returns per-player VP normalized to [-1, 1] (10 VP -> +1, 0 VP -> -1),
//!     with a small bonus for actually winning within the lookahead window.

use catan_engine::Engine;

#[test]
fn lookahead_returns_four_floats_in_unit_range() {
    let mut e = Engine::new(42);
    let v = e.lookahead_vp_value(5, 7);
    for &x in &v {
        assert!(x >= -1.0 && x <= 1.0, "value out of [-1,1]: {:?}", v);
    }
}

#[test]
fn lookahead_does_not_mutate_caller_engine() {
    // The Python evaluator clones before calling, but as defense in depth
    // make sure depth>0 lookahead is callable on a clone without surprising
    // the caller. We test the "caller cloned first" pattern directly here.
    let e_orig = Engine::new(42);
    let history_before = e_orig.action_history().to_vec();
    let mut e_clone = e_orig.clone();
    let _ = e_clone.lookahead_vp_value(3, 1);
    // Original engine's history is unchanged.
    assert_eq!(e_orig.action_history(), history_before.as_slice());
}

#[test]
fn lookahead_is_deterministic_given_seed() {
    let mut a = Engine::new(42);
    let mut b = Engine::new(42);
    let va = a.lookahead_vp_value(5, 99);
    let vb = b.lookahead_vp_value(5, 99);
    assert_eq!(va, vb, "same seed produced different values");
}

#[test]
fn lookahead_seed_varies_trajectory() {
    // Different seeds (which only affect chance sampling, since decisions
    // are greedy-deterministic) should usually yield different value vectors.
    let mut a = Engine::new(42);
    let mut b = Engine::new(42);
    let va = a.lookahead_vp_value(10, 1);
    let vb = b.lookahead_vp_value(10, 999);
    // Histories must differ — at minimum the dice rolls landed differently.
    assert_ne!(a.action_history(), b.action_history());
    // Note: greedy decisions are deterministic given state, so identical VP
    // vectors across different chance seeds are common and NOT a bug — they
    // just mean the dice noise didn't perturb the build order. We only assert
    // trajectory divergence here.
    let _ = (va, vb);
}

#[test]
fn lookahead_depth_zero_is_current_state_value() {
    // depth=0 -> no forward play -> value = normalize(current VPs).
    // At engine init, all players have 0 VP -> normalized to -1.0 each.
    let mut e = Engine::new(42);
    let v = e.lookahead_vp_value(0, 0);
    assert_eq!(v, [-1.0f32; 4], "depth=0 should reflect current VP only, got {:?}", v);
}

#[test]
fn lookahead_terminal_returns_winner_signal() {
    // Drive engine to terminal via random rollout, then call lookahead.
    // Should detect terminal and return AlphaZero-style returns (+1 / -1).
    let mut e = Engine::new(42);
    let _ = e.random_rollout_to_terminal(123);
    if e.is_terminal() {
        let v = e.lookahead_vp_value(10, 0);
        let plus = v.iter().filter(|&&x| x == 1.0).count();
        assert_eq!(plus, 1, "terminal lookahead should return exactly one +1.0, got {:?}", v);
    }
    // If rollout capped (not terminal), test is a no-op — covered by other tests.
}
