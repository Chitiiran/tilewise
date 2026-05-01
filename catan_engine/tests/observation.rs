use catan_engine::observation::{build_observation, N_SCALARS};
use catan_engine::Engine;

#[test]
fn observation_shapes_match_spec() {
    let engine = Engine::new(42);
    let obs = build_observation(&engine.state, engine.state.current_player);
    assert_eq!(obs.hex_features.len(), 19 * catan_engine::observation::F_HEX);
    assert_eq!(obs.vertex_features.len(), 54 * catan_engine::observation::F_VERT);
    assert_eq!(obs.edge_features.len(), 72 * catan_engine::observation::F_EDGE);
    assert_eq!(obs.legal_mask.len(), catan_engine::actions::ACTION_SPACE_SIZE);
}

#[test]
fn legal_mask_count_matches_legal_actions() {
    let engine = Engine::new(42);
    let obs = build_observation(&engine.state, engine.state.current_player);
    let mask_true = obs.legal_mask.iter().filter(|&&b| b).count();
    assert_eq!(mask_true, engine.legal_actions().len());
}

#[test]
fn scalars_layout_v2_size_is_59() {
    // v2 expanded scalars: see observation.rs SCALAR_LAYOUT comment.
    assert_eq!(N_SCALARS, 59);
    let engine = Engine::new(42);
    let obs = build_observation(&engine.state, engine.state.current_player);
    assert_eq!(obs.scalars.len(), N_SCALARS);
}

#[test]
fn scalars_bank_section_reflects_initial_bank() {
    // After fresh new(): bank is full (19 of each); fully normalized to 1.0.
    let engine = Engine::new(42);
    let obs = build_observation(&engine.state, engine.state.current_player);
    // Bank section is the last 5 scalars.
    let n = obs.scalars.len();
    for r in 0..5 {
        let v = obs.scalars[n - 5 + r];
        assert!((v - 1.0).abs() < 1e-6, "bank[{r}] = {v}, expected 1.0");
    }
}

#[test]
fn scalars_dev_cards_held_zero_at_start() {
    // No one has any dev cards at initial state.
    let engine = Engine::new(42);
    let obs = build_observation(&engine.state, engine.state.current_player);
    // viewer dev cards held block: 5 scalars. Located at offset 21 (see SCALAR_LAYOUT).
    for k in 0..5 {
        assert_eq!(obs.scalars[21 + k], 0.0);
    }
}

#[test]
fn scalars_perspective_rotation_consistent() {
    // VP block is perspective-rotated. From player 0's viewpoint at index 8,
    // and from player 1's at index 8+0 should be player 1's VP itself.
    let engine = Engine::new(42);
    let obs0 = build_observation(&engine.state, 0);
    let obs1 = build_observation(&engine.state, 1);
    // VP[viewer] is at index 8 in scalars. After Setup, all four are 0.
    assert_eq!(obs0.scalars[8], engine.state.vp[0] as f32);
    assert_eq!(obs1.scalars[8], engine.state.vp[1] as f32);
}

#[test]
fn scalars_uses_cached_legal_mask_when_available() {
    // Computing observation should not recompute legal_actions if cache is fresh.
    // Mostly a smoke test: the count matches between fresh and cached lookups.
    let mut engine = Engine::new(42);
    let mask = engine.legal_mask();  // populates cache
    let cached_count = mask.iter_ids().count();
    let obs = build_observation(&engine.state, engine.state.current_player);
    let obs_count = obs.legal_mask.iter().filter(|&&b| b).count();
    assert_eq!(cached_count, obs_count);
}
