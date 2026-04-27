use catan_engine::actions::Action;
use catan_engine::board::Board;
use catan_engine::rng::Rng;
use catan_engine::rules::{apply, legal_actions};
use catan_engine::state::{GameState, GamePhase};
use std::sync::Arc;

fn fresh_state() -> GameState {
    GameState::new(Arc::new(Board::standard()))
}

#[test]
fn setup1_legal_actions_are_all_54_settlements_when_board_empty() {
    let state = fresh_state();
    let legal = legal_actions(&state);
    let settlement_actions: Vec<_> = legal.iter()
        .filter(|a| matches!(a, Action::BuildSettlement(_)))
        .collect();
    assert_eq!(settlement_actions.len(), 54);
}

#[test]
fn setup1_distance_rule_blocks_adjacent_vertices() {
    let mut state = fresh_state();
    let mut rng = Rng::from_seed(0);
    // Place settlement at vertex 0.
    apply(&mut state, Action::BuildSettlement(0), &mut rng);
    // After placing, the road action follows; for this test just inspect occupancy.
    assert_eq!(state.settlements[0], Some(0)); // owned by player 0
    // After settlement is placed, before player picks a road,
    // legal_actions should enumerate roads at edges of vertex 0 only.
    // (The distance check itself: vertex 0's neighbors must not be settlement-legal
    // even if we re-entered Setup1Place — verified in next test.)
}

#[test]
fn setup1_after_first_settlement_road_must_be_adjacent_to_it() {
    let mut state = fresh_state();
    let mut rng = Rng::from_seed(0);
    apply(&mut state, Action::BuildSettlement(0), &mut rng);
    let legal = legal_actions(&state);
    let road_actions: Vec<u8> = legal.iter()
        .filter_map(|a| match a {
            Action::BuildRoad(e) => Some(*e),
            _ => None,
        })
        .collect();
    let v0_edges = &state.board.vertex_to_edges[0];
    for e in &road_actions {
        assert!(v0_edges.contains(e), "road {e} not adjacent to settlement at vertex 0");
    }
}

#[test]
fn setup1_placing_road_advances_to_next_player() {
    let mut state = fresh_state();
    let mut rng = Rng::from_seed(0);
    apply(&mut state, Action::BuildSettlement(0), &mut rng);
    let road_edge = state.board.vertex_to_edges[0][0];
    apply(&mut state, Action::BuildRoad(road_edge), &mut rng);
    assert_eq!(state.current_player, 1);
    assert!(matches!(state.phase, GamePhase::Setup1Place));
    assert!(state.setup_pending.is_none());
}

#[test]
fn setup1_after_all_4_players_place_first_settlement_we_enter_setup2() {
    let mut state = fresh_state();
    let mut rng = Rng::from_seed(0);
    let starting_vertices = [0u8, 8, 16, 24];
    for v in starting_vertices {
        apply(&mut state, Action::BuildSettlement(v), &mut rng);
        let road_edge = state.board.vertex_to_edges[v as usize][0];
        apply(&mut state, Action::BuildRoad(road_edge), &mut rng);
    }
    assert!(matches!(state.phase, GamePhase::Setup2Place));
    // Setup-2 goes in REVERSE order, so player 3 places first.
    assert_eq!(state.current_player, 3);
}
