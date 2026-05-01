//! Phase 2.4: player-to-player 1-for-1 trade tests.
//!
//! Engine resolves a ProposeTrade by iterating opponents in seat order; first
//! opponent with ≥1 of the requested resource accepts. If none, no resources move.

use catan_engine::actions::Action;
use catan_engine::board::{Board, Resource};
use catan_engine::rng::Rng;
use catan_engine::rules::{apply, legal_actions};
use catan_engine::state::{GameState, GamePhase};
use std::sync::Arc;

fn fresh() -> GameState {
    let mut state = GameState::new(Arc::new(Board::standard()));
    state.phase = GamePhase::Main;
    state.current_player = 0;
    state
}

#[test]
fn propose_trade_legal_when_proposer_has_give_and_opp_has_get() {
    let mut state = fresh();
    state.hands[0] = [1, 0, 0, 0, 0]; // P0 has 1 wood
    state.hands[1] = [0, 1, 0, 0, 0]; // P1 has 1 brick
    let legal = legal_actions(&state);
    let trades: Vec<_> = legal.iter().filter_map(|a| match a {
        Action::ProposeTrade { give, get } => Some((*give, *get)),
        _ => None,
    }).collect();
    assert!(trades.contains(&(Resource::Wood, Resource::Brick)));
}

#[test]
fn propose_trade_not_legal_if_proposer_lacks_give() {
    let mut state = fresh();
    state.hands[0] = [0, 0, 0, 0, 0]; // P0 has nothing
    state.hands[1] = [1, 0, 0, 0, 0]; // P1 has wood
    let legal = legal_actions(&state);
    assert!(!legal.iter().any(|a| matches!(a, Action::ProposeTrade { .. })));
}

#[test]
fn propose_trade_not_legal_if_no_opponent_has_get() {
    let mut state = fresh();
    state.hands[0] = [1, 0, 0, 0, 0];
    // No opponent has any resource.
    let legal = legal_actions(&state);
    assert!(!legal.iter().any(|a| matches!(a, Action::ProposeTrade { .. })));
}

#[test]
fn propose_trade_executes_with_first_seat_order_acceptor() {
    let mut state = fresh();
    state.hands[0] = [1, 0, 0, 0, 0]; // P0 wants brick, has wood
    state.hands[1] = [0, 0, 0, 0, 0]; // P1 has nothing — can't accept
    state.hands[2] = [0, 1, 0, 0, 0]; // P2 has brick — should accept
    state.hands[3] = [0, 1, 0, 0, 0]; // P3 also has brick — won't be reached
    let mut rng = Rng::from_seed(0);
    apply(&mut state, Action::ProposeTrade { give: Resource::Wood, get: Resource::Brick }, &mut rng);
    // P0: -wood, +brick
    assert_eq!(state.hands[0][0], 0);
    assert_eq!(state.hands[0][1], 1);
    // P2: -brick, +wood
    assert_eq!(state.hands[2][0], 1);
    assert_eq!(state.hands[2][1], 0);
    // P3: untouched
    assert_eq!(state.hands[3][0], 0);
    assert_eq!(state.hands[3][1], 1);
}

#[test]
fn propose_trade_no_movement_if_no_acceptor() {
    let mut state = fresh();
    state.hands[0] = [1, 0, 0, 0, 0]; // P0 has wood
    state.hands[1] = [1, 0, 0, 0, 0]; // P1 has wood (not what P0 wants)
    state.hands[2] = [1, 0, 0, 0, 0];
    state.hands[3] = [1, 0, 0, 0, 0];
    let mut rng = Rng::from_seed(0);
    // No one has brick — but legal_actions would have already filtered this
    // out. We bypass legality and apply directly to confirm engine doesn't
    // crash if called with an unfulfillable trade.
    apply(&mut state, Action::ProposeTrade { give: Resource::Wood, get: Resource::Brick }, &mut rng);
    // No movement.
    assert_eq!(state.hands[0][0], 1);
    assert_eq!(state.hands[0][1], 0);
}

#[test]
fn propose_trade_resource_conservation() {
    let mut state = fresh();
    state.hands[0] = [2, 0, 0, 0, 0];
    state.hands[1] = [0, 3, 0, 0, 0];
    // Adjust bank for the resources we placed in hands.
    state.bank[0] -= 2;
    state.bank[1] -= 3;
    let mut rng = Rng::from_seed(0);
    apply(&mut state, Action::ProposeTrade { give: Resource::Wood, get: Resource::Brick }, &mut rng);
    // Resource invariant: each resource sums to 19 across hands+bank.
    for r in 0..5 {
        let total: u32 = (0..4).map(|p| state.hands[p][r] as u32).sum::<u32>()
                           + state.bank[r] as u32;
        assert_eq!(total, 19, "resource {} broken", r);
    }
}

#[test]
fn propose_trade_seat_order_starts_with_next_player() {
    // Verify offset starts at 1, then 2, then 3 — next-player first.
    let mut state = fresh();
    state.current_player = 2;
    state.hands[2] = [1, 0, 0, 0, 0];
    state.hands[3] = [0, 1, 0, 0, 0]; // next in seat order — should accept
    state.hands[0] = [0, 1, 0, 0, 0];
    state.hands[1] = [0, 1, 0, 0, 0];
    let mut rng = Rng::from_seed(0);
    apply(&mut state, Action::ProposeTrade { give: Resource::Wood, get: Resource::Brick }, &mut rng);
    // P3 should have accepted (next after P2).
    assert_eq!(state.hands[3][0], 1, "P3 (next in seat order) should have accepted");
    assert_eq!(state.hands[3][1], 0);
    // P0 and P1 untouched.
    assert_eq!(state.hands[0][1], 1);
    assert_eq!(state.hands[1][1], 1);
}
