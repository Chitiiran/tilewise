use catan_engine::actions::Action;
use catan_engine::board::{Board, Resource};
use catan_engine::rng::Rng;
use catan_engine::rules::{apply, legal_actions};
use catan_engine::state::{GameState, GamePhase};
use std::sync::Arc;

#[test]
fn legal_discard_actions_are_resources_player_owns() {
    let mut state = GameState::new(Arc::new(Board::standard()));
    state.phase = GamePhase::Discard { remaining: [3, 0, 0, 0] };
    state.current_player = 0;
    state.hands[0] = [3, 0, 5, 0, 0]; // 3 wood, 5 sheep
    let legal = legal_actions(&state);
    let discarded: Vec<Resource> = legal.iter()
        .filter_map(|a| if let Action::Discard(r) = a { Some(*r) } else { None })
        .collect();
    assert!(discarded.contains(&Resource::Wood));
    assert!(discarded.contains(&Resource::Sheep));
    assert!(!discarded.contains(&Resource::Brick));
}

#[test]
fn discarding_decrements_hand_and_remaining() {
    let mut state = GameState::new(Arc::new(Board::standard()));
    state.phase = GamePhase::Discard { remaining: [2, 0, 0, 0] };
    state.current_player = 0;
    state.hands[0] = [5, 0, 0, 0, 0];
    let mut rng = Rng::from_seed(0);
    apply(&mut state, Action::Discard(Resource::Wood), &mut rng);
    assert_eq!(state.hands[0][Resource::Wood as usize], 4);
    if let GamePhase::Discard { remaining } = state.phase {
        assert_eq!(remaining[0], 1);
    } else { panic!("expected Discard phase"); }
}

#[test]
fn last_discard_transitions_to_move_robber() {
    let mut state = GameState::new(Arc::new(Board::standard()));
    state.phase = GamePhase::Discard { remaining: [1, 0, 0, 0] };
    state.current_player = 0; // current_player is the roller, who triggers the robber move
    state.hands[0] = [1, 0, 0, 0, 0];
    let mut rng = Rng::from_seed(0);
    apply(&mut state, Action::Discard(Resource::Wood), &mut rng);
    assert!(matches!(state.phase, GamePhase::MoveRobber));
}
