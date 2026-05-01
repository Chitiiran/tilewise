//! Phase 2.3: dev card behavior tests + Phase 2.2 largest army.

use catan_engine::actions::Action;
use catan_engine::board::{Board, Resource};
use catan_engine::rng::Rng;
use catan_engine::rules::{apply, legal_actions, update_largest_army};
use catan_engine::state::{
    GameState, GamePhase,
    DEV_CARD_KNIGHT, DEV_CARD_ROAD_BUILDING, DEV_CARD_MONOPOLY, DEV_CARD_YOP, DEV_CARD_VP,
    DEV_CARD_DECK_STANDARD, N_DEV_CARDS,
};
use std::sync::Arc;

fn fresh() -> GameState {
    let mut state = GameState::new(Arc::new(Board::standard()));
    state.phase = GamePhase::Main;
    state.current_player = 0;
    state
}

#[test]
fn buy_dev_card_costs_1s_1wh_1o_and_draws() {
    let mut state = fresh();
    state.hands[0] = [0, 0, 1, 1, 1]; // 1 sheep, 1 wheat, 1 ore — exact cost
    state.bank[2] -= 1; state.bank[3] -= 1; state.bank[4] -= 1;
    let mut rng = Rng::from_seed(42);
    let cards_before: u8 = state.dev_cards_held[0].iter().sum::<u8>() + state.dev_cards_played[0].iter().sum::<u8>();
    apply(&mut state, Action::BuyDevCard, &mut rng);
    let cards_after: u8 = state.dev_cards_held[0].iter().sum::<u8>() + state.dev_cards_played[0].iter().sum::<u8>();
    // VP cards apply immediately (counted in dev_cards_played); others land in held.
    assert_eq!(cards_after, cards_before + 1, "exactly 1 dev card should have been drawn");
    // Resources spent.
    assert_eq!(state.hands[0][2], 0); // sheep
    assert_eq!(state.hands[0][3], 0);
    assert_eq!(state.hands[0][4], 0);
}

#[test]
fn buy_dev_card_not_legal_without_resources() {
    let mut state = fresh();
    state.hands[0] = [0, 0, 0, 1, 1]; // missing sheep
    let legal = legal_actions(&state);
    assert!(!legal.iter().any(|a| matches!(a, Action::BuyDevCard)));
}

#[test]
fn cant_play_dev_card_on_turn_bought() {
    let mut state = fresh();
    state.dev_cards_held[0][DEV_CARD_KNIGHT] = 1;
    state.dev_cards_just_bought[0] = true;
    let legal = legal_actions(&state);
    assert!(!legal.iter().any(|a| matches!(a, Action::PlayKnight)));
}

#[test]
fn play_knight_increments_largest_army_count() {
    let mut state = fresh();
    state.dev_cards_held[0][DEV_CARD_KNIGHT] = 1;
    let mut rng = Rng::from_seed(0);
    apply(&mut state, Action::PlayKnight, &mut rng);
    assert_eq!(state.knights_played[0], 1);
    assert_eq!(state.dev_cards_held[0][DEV_CARD_KNIGHT], 0);
    assert!(matches!(state.phase, GamePhase::MoveRobber));
    assert!(state.dev_card_played_this_turn[0]);
}

#[test]
fn largest_army_kicks_in_at_3_knights() {
    let mut state = fresh();
    state.knights_played[0] = 3;
    update_largest_army(&mut state);
    assert_eq!(state.largest_army_holder, Some(0));
    assert_eq!(state.vp[0], 2, "+2 VP for largest army");
}

#[test]
fn largest_army_below_3_no_holder() {
    let mut state = fresh();
    state.knights_played[0] = 2;
    update_largest_army(&mut state);
    assert!(state.largest_army_holder.is_none());
    assert_eq!(state.vp[0], 0);
}

#[test]
fn largest_army_transfers_to_strict_max() {
    let mut state = fresh();
    state.knights_played[0] = 3;
    update_largest_army(&mut state);
    assert_eq!(state.largest_army_holder, Some(0));
    assert_eq!(state.vp[0], 2);

    // P1 catches up — tie, no transfer (prior holder keeps).
    state.knights_played[1] = 3;
    update_largest_army(&mut state);
    assert_eq!(state.largest_army_holder, Some(0));
    assert_eq!(state.vp[0], 2);
    assert_eq!(state.vp[1], 0);

    // P1 surpasses.
    state.knights_played[1] = 4;
    update_largest_army(&mut state);
    assert_eq!(state.largest_army_holder, Some(1));
    assert_eq!(state.vp[0], 0, "P0 lost the +2 VP");
    assert_eq!(state.vp[1], 2);
}

#[test]
fn play_monopoly_takes_resource_from_all_opponents() {
    let mut state = fresh();
    state.dev_cards_held[0][DEV_CARD_MONOPOLY] = 1;
    state.hands[0] = [1, 0, 0, 0, 0];
    state.hands[1] = [3, 0, 0, 0, 0];
    state.hands[2] = [5, 0, 0, 0, 0];
    state.hands[3] = [2, 0, 0, 0, 0];
    let mut rng = Rng::from_seed(0);
    apply(&mut state, Action::PlayMonopoly(Resource::Wood), &mut rng);
    assert_eq!(state.hands[0][0], 1 + 3 + 5 + 2);
    assert_eq!(state.hands[1][0], 0);
    assert_eq!(state.hands[2][0], 0);
    assert_eq!(state.hands[3][0], 0);
}

#[test]
fn play_year_of_plenty_takes_2_from_bank() {
    let mut state = fresh();
    state.dev_cards_held[0][DEV_CARD_YOP] = 1;
    let bank_wood_before = state.bank[0];
    let bank_brick_before = state.bank[1];
    let mut rng = Rng::from_seed(0);
    apply(&mut state, Action::PlayYearOfPlenty(Resource::Wood, Resource::Brick), &mut rng);
    assert_eq!(state.hands[0][0], 1);
    assert_eq!(state.hands[0][1], 1);
    assert_eq!(state.bank[0], bank_wood_before - 1);
    assert_eq!(state.bank[1], bank_brick_before - 1);
}

#[test]
fn play_road_building_grants_resources() {
    let mut state = fresh();
    state.dev_cards_held[0][DEV_CARD_ROAD_BUILDING] = 1;
    let mut rng = Rng::from_seed(0);
    apply(&mut state, Action::PlayRoadBuilding, &mut rng);
    // Player gained 2 wood + 2 brick (assuming bank had them).
    assert_eq!(state.hands[0][0], 2);
    assert_eq!(state.hands[0][1], 2);
}

#[test]
fn dev_deck_standard_is_25_cards() {
    let total: u8 = DEV_CARD_DECK_STANDARD.iter().sum();
    assert_eq!(total, 25);
    assert_eq!(DEV_CARD_DECK_STANDARD[DEV_CARD_KNIGHT], 14);
    assert_eq!(DEV_CARD_DECK_STANDARD[DEV_CARD_VP], 5);
}

#[test]
fn dev_card_played_this_turn_resets_on_end_turn() {
    let mut state = fresh();
    state.dev_cards_held[0][DEV_CARD_YOP] = 1;
    let mut rng = Rng::from_seed(0);
    apply(&mut state, Action::PlayYearOfPlenty(Resource::Wood, Resource::Brick), &mut rng);
    assert!(state.dev_card_played_this_turn[0]);
    apply(&mut state, Action::EndTurn, &mut rng);
    assert!(!state.dev_card_played_this_turn[0]);
}
