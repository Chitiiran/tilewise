use catan_engine::observation::{build_observation, F_VERT, N_SCALARS};
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
fn vertex_features_widened_to_13_for_port_kind() {
    // P3: vertex_features now carries [empty, settle, city, owner0..3, port_generic,
    // port_wood, port_brick, port_sheep, port_wheat, port_ore] = 13 dims.
    assert_eq!(F_VERT, 13);
    let engine = Engine::new(42);
    let obs = build_observation(&engine.state, 0);
    assert_eq!(obs.vertex_features.len(), 54 * F_VERT);
}

#[test]
fn vertex_features_set_port_flags_for_port_vertices() {
    // Each of the 9 ports occupies 2 vertices. After P2 the canonical layout
    // is fixed; we pick port vertex 0 (generic 3:1) and port vertex 11 (sheep)
    // and assert the right one-hot bit is set in their feature row.
    let engine = Engine::new(42);
    let obs = build_observation(&engine.state, 0);
    // Port-kind one-hot starts at offset 7 within F_VERT.
    // Layout: [generic, wood, brick, sheep, wheat, ore]
    let port_kind_offset = 7;
    // Vertex 0 is on the (0,4) generic 3:1 port.
    let v0 = 0;
    assert_eq!(obs.vertex_features[v0 * F_VERT + port_kind_offset + 0], 1.0,
        "vertex 0 should have generic-port flag set");
    // Vertex 11 is on the (11,16) sheep 2:1 port.
    let v11 = 11;
    assert_eq!(obs.vertex_features[v11 * F_VERT + port_kind_offset + 3], 1.0,
        "vertex 11 should have sheep-port flag set (offset 7+3)");
    // Vertex 8 is interior (no port).
    let v8 = 8;
    for k in 0..6 {
        assert_eq!(obs.vertex_features[v8 * F_VERT + port_kind_offset + k], 0.0,
            "interior vertex 8 should have no port flags set (k={})", k);
    }
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
