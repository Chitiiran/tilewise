use catan_engine::observation::build_observation;
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
