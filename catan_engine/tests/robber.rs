use catan_engine::actions::Action;
use catan_engine::board::Board;
use catan_engine::rng::Rng;
use catan_engine::rules::{apply, legal_actions};
use catan_engine::state::{GameState, GamePhase};
use std::sync::Arc;

#[test]
fn legal_move_robber_actions_exclude_current_hex() {
    let mut state = GameState::new(Arc::new(Board::standard()));
    state.phase = GamePhase::MoveRobber;
    let robber = state.robber_hex;
    let legal = legal_actions(&state);
    let hexes: Vec<u8> = legal.iter()
        .filter_map(|a| if let Action::MoveRobber(h) = a { Some(*h) } else { None })
        .collect();
    assert_eq!(hexes.len(), 18);
    assert!(!hexes.contains(&robber));
}

#[test]
fn moving_robber_to_unoccupied_hex_skips_steal() {
    let mut state = GameState::new(Arc::new(Board::standard()));
    state.phase = GamePhase::MoveRobber;
    state.current_player = 0;
    let mut rng = Rng::from_seed(0);
    let target = (0..19u8).find(|&h| h != state.robber_hex).unwrap();
    apply(&mut state, Action::MoveRobber(target), &mut rng);
    assert_eq!(state.robber_hex, target);
    assert!(matches!(state.phase, GamePhase::Main));
}

#[test]
fn moving_robber_to_hex_with_only_self_buildings_skips_steal() {
    let mut state = GameState::new(Arc::new(Board::standard()));
    // Put player 0's settlement adjacent to hex 5; nobody else has buildings there.
    let target_hex = 5u8;
    let v = state.board.hex_to_vertices[target_hex as usize][0];
    state.settlements[v as usize] = Some(0);
    state.phase = GamePhase::MoveRobber;
    state.current_player = 0;
    let mut rng = Rng::from_seed(0);
    apply(&mut state, Action::MoveRobber(target_hex), &mut rng);
    assert!(matches!(state.phase, GamePhase::Main));
}
