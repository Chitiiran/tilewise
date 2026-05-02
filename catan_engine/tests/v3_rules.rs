//! v3 rule-flag tests — `Engine::with_rules(seed, vp_target, bonuses)`.
//!
//! v3 is full Catan with two reversible flags: a configurable winning-VP
//! threshold (default 10) and a switch that disables the +2 longest-road and
//! largest-army bonuses (default on). The flags exist so the same engine
//! binary can run v2 (full rules) and v3 (5-VP, no bonuses) self-play / MCTS
//! / training without rebuilding.
//!
//! These tests pin the contract:
//!   * Defaults match v2 (`Engine::new` is unchanged).
//!   * `vp_target` actually controls the win threshold.
//!   * When `bonuses=false`, longest-road and largest-army still TRACK
//!     (length, holder, knights), but never award the +2 VP.
//!   * The flags participate in serialization where applicable.

use catan_engine::engine::Engine;
use catan_engine::state::WIN_VP;

#[test]
fn engine_new_uses_v2_defaults() {
    let e = Engine::new(42);
    assert_eq!(e.state.vp_target, WIN_VP, "Engine::new should default to WIN_VP=10");
    assert!(e.state.bonuses_enabled, "Engine::new should default to bonuses on");
}

#[test]
fn with_rules_stores_flags() {
    let e = Engine::with_rules(42, 5, false);
    assert_eq!(e.state.vp_target, 5);
    assert!(!e.state.bonuses_enabled);

    let e2 = Engine::with_rules(42, 7, true);
    assert_eq!(e2.state.vp_target, 7);
    assert!(e2.state.bonuses_enabled);
}

#[test]
fn vp_target_controls_win_threshold_default() {
    // With v2 default vp_target=10, hitting 9 VP doesn't end the game.
    let mut e = Engine::new(0);
    e.state.vp[0] = 9;
    catan_engine::rules::check_win_for_test(&mut e.state);
    assert!(!e.is_terminal(), "9 VP should not end a vp_target=10 game");

    e.state.vp[0] = 10;
    catan_engine::rules::check_win_for_test(&mut e.state);
    assert!(e.is_terminal(), "10 VP should end a vp_target=10 game");
}

#[test]
fn vp_target_controls_win_threshold_v3() {
    // With v3 vp_target=5, hitting 5 VP ends the game; 4 doesn't.
    let mut e = Engine::with_rules(0, 5, false);
    e.state.vp[0] = 4;
    catan_engine::rules::check_win_for_test(&mut e.state);
    assert!(!e.is_terminal(), "4 VP should not end a vp_target=5 game");

    e.state.vp[0] = 5;
    catan_engine::rules::check_win_for_test(&mut e.state);
    assert!(e.is_terminal(), "5 VP should end a vp_target=5 game");
}

#[test]
fn longest_road_no_vp_when_bonuses_off() {
    // Force the holder + length state directly and verify VP doesn't move
    // when bonuses are disabled.
    let mut e = Engine::with_rules(0, 5, false);
    // Pretend P0 should be the holder per the length array, with no prior
    // holder. With bonuses=false, vp[0] must stay at 0.
    e.state.longest_road_length = [6, 0, 0, 0];
    e.state.longest_road_holder = None;
    e.state.vp = [0, 0, 0, 0];
    // Manually invoke the holder-transfer path (without recomputing length
    // from roads, which would zero our forced array).
    catan_engine::rules::transfer_longest_road_holder_for_test(&mut e.state);
    assert_eq!(
        e.state.vp[0], 0,
        "bonuses=false: holder may transfer but no +2 VP"
    );
}

#[test]
fn longest_road_grants_vp_when_bonuses_on() {
    // Same setup but with bonuses=true — the +2 must be awarded.
    let mut e = Engine::with_rules(0, 10, true);
    e.state.longest_road_length = [6, 0, 0, 0];
    e.state.longest_road_holder = None;
    e.state.vp = [0, 0, 0, 0];
    catan_engine::rules::transfer_longest_road_holder_for_test(&mut e.state);
    assert_eq!(e.state.vp[0], 2, "bonuses=true: holder gets +2 VP");
    assert_eq!(e.state.longest_road_holder, Some(0));
}

#[test]
fn largest_army_no_vp_when_bonuses_off() {
    let mut e = Engine::with_rules(0, 5, false);
    e.state.knights_played = [3, 0, 0, 0];
    e.state.largest_army_holder = None;
    e.state.vp = [0, 0, 0, 0];
    catan_engine::rules::update_largest_army(&mut e.state);
    assert_eq!(
        e.state.vp[0], 0,
        "bonuses=false: largest_army holder gets no +2 VP"
    );
    assert_eq!(
        e.state.largest_army_holder,
        Some(0),
        "holder still tracked even without VP award"
    );
}

#[test]
fn largest_army_grants_vp_when_bonuses_on() {
    let mut e = Engine::with_rules(0, 10, true);
    e.state.knights_played = [3, 0, 0, 0];
    e.state.largest_army_holder = None;
    e.state.vp = [0, 0, 0, 0];
    catan_engine::rules::update_largest_army(&mut e.state);
    assert_eq!(e.state.vp[0], 2, "bonuses=true: largest army holder gets +2 VP");
    assert_eq!(e.state.largest_army_holder, Some(0));
}

#[test]
fn bonuses_off_does_not_remove_existing_vp_on_holder_loss() {
    // If bonuses=false, the engine never granted +2 in the first place,
    // so when a holder transfers or loses status, no VP is subtracted.
    let mut e = Engine::with_rules(0, 5, false);
    // Simulate a state where P0 was previously the holder somehow.
    // Under v3, this should never have happened, but test the saturating
    // arithmetic is safe regardless.
    e.state.knights_played = [2, 3, 0, 0]; // P1 takes over
    e.state.largest_army_holder = Some(0); // stale
    e.state.vp = [0, 0, 0, 0]; // no +2 was ever granted
    catan_engine::rules::update_largest_army(&mut e.state);
    assert_eq!(e.state.vp[0], 0, "P0 had 0 VP; saturating_sub keeps 0");
    assert_eq!(e.state.vp[1], 0, "P1 takes holder but bonuses=false → no +2");
    assert_eq!(e.state.largest_army_holder, Some(1));
}
