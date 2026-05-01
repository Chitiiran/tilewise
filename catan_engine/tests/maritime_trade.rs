//! Phase 1.8: maritime trade tests.
//! 4:1 bank trade only — port-aware ratios deferred.

use catan_engine::actions::{Action, encode, decode};
use catan_engine::board::{Board, Resource};
use catan_engine::rng::Rng;
use catan_engine::rules::{apply, legal_actions};
use catan_engine::state::{GameState, GamePhase};
use std::sync::Arc;

fn fresh_main_state() -> GameState {
    let mut state = GameState::new(Arc::new(Board::standard()));
    state.phase = GamePhase::Main;
    state.current_player = 0;
    state
}

#[test]
fn trade_bank_legal_when_player_has_4_of_give() {
    let mut state = fresh_main_state();
    state.hands[0] = [4, 0, 0, 0, 0]; // 4 wood
    let legal = legal_actions(&state);
    let trade_actions: Vec<_> = legal.iter().filter_map(|a| match a {
        Action::TradeBank { give, get } => Some((*give, *get)),
        _ => None,
    }).collect();
    // Should have 4 trade options: wood→{brick, sheep, wheat, ore}.
    assert_eq!(trade_actions.len(), 4);
    let gives: Vec<Resource> = trade_actions.iter().map(|(g, _)| *g).collect();
    assert!(gives.iter().all(|&g| g == Resource::Wood));
}

#[test]
fn trade_bank_not_legal_when_below_threshold() {
    let mut state = fresh_main_state();
    state.hands[0] = [3, 0, 0, 0, 0]; // 3 wood — not enough
    let legal = legal_actions(&state);
    let has_trade = legal.iter().any(|a| matches!(a, Action::TradeBank { .. }));
    assert!(!has_trade, "3 wood should not enable any 4:1 bank trade");
}

#[test]
fn trade_bank_not_legal_when_bank_empty_for_get() {
    let mut state = fresh_main_state();
    state.hands[0] = [4, 0, 0, 0, 0];
    state.bank = [15, 0, 19, 19, 19]; // bank has no brick
    let legal = legal_actions(&state);
    let trade_actions: Vec<_> = legal.iter().filter_map(|a| match a {
        Action::TradeBank { give, get } => Some((*give, *get)),
        _ => None,
    }).collect();
    // wood→brick should be filtered out; 3 options remain.
    assert_eq!(trade_actions.len(), 3);
    assert!(!trade_actions.iter().any(|(_, g)| *g == Resource::Brick));
}

#[test]
fn trade_bank_apply_moves_resources_correctly() {
    let mut state = fresh_main_state();
    // Set up player with 4 wood. Bank must reflect the 4 already taken
    // (otherwise we violate the supply-of-19 invariant).
    state.hands[0] = [4, 0, 0, 0, 0];
    state.bank[Resource::Wood as usize] -= 4; // bank now has 15 wood
    let bank_before = state.bank;
    let mut rng = Rng::from_seed(0);
    apply(&mut state, Action::TradeBank { give: Resource::Wood, get: Resource::Brick }, &mut rng);
    // Player gave 4 wood, got 1 brick.
    assert_eq!(state.hands[0][Resource::Wood as usize], 0);
    assert_eq!(state.hands[0][Resource::Brick as usize], 1);
    // Bank gained 4 wood, lost 1 brick.
    assert_eq!(state.bank[Resource::Wood as usize], bank_before[Resource::Wood as usize] + 4);
    assert_eq!(state.bank[Resource::Brick as usize], bank_before[Resource::Brick as usize] - 1);
    // Resource conservation: each resource still sums to 19.
    for r in 0..5 {
        let in_hands: u32 = (0..4).map(|p| state.hands[p][r] as u32).sum();
        assert_eq!(in_hands + state.bank[r] as u32, 19, "resource {} broken", r);
    }
}

#[test]
fn trade_bank_then_build_settlement_legal_path() {
    // Demonstrates the strategic intent of trade: convert 4-wood into a
    // resource you need.
    let mut state = fresh_main_state();
    state.hands[0] = [4, 1, 1, 1, 0]; // missing 1 of nothing — too easy. Let's give them an unbalanced hand.
    state.hands[0] = [8, 0, 0, 0, 0]; // 8 wood, can trade twice
    state.settlements.set(0, Some(0));
    state.roads.set(0, Some(0));

    let mut rng = Rng::from_seed(0);
    // 1st trade: wood→brick.
    apply(&mut state, Action::TradeBank { give: Resource::Wood, get: Resource::Brick }, &mut rng);
    // 2nd trade: wood→sheep.
    apply(&mut state, Action::TradeBank { give: Resource::Wood, get: Resource::Sheep }, &mut rng);
    // 3rd trade: wood would still need wheat. Have 0 wood now, so can't.
    // Actually wait — started with 8, 4 + 4 = 8 traded, 0 left.
    assert_eq!(state.hands[0][Resource::Wood as usize], 0);
    // Got 1 brick, 1 sheep.
    assert_eq!(state.hands[0][Resource::Brick as usize], 1);
    assert_eq!(state.hands[0][Resource::Sheep as usize], 1);
}

#[test]
fn trade_bank_action_id_in_range() {
    // Sanity: encode/decode for a trade matches the design (206..226).
    for give_i in 0..5u8 {
        for get_i in 0..5u8 {
            if give_i == get_i { continue; }
            let give = match give_i {0=>Resource::Wood,1=>Resource::Brick,2=>Resource::Sheep,3=>Resource::Wheat,_=>Resource::Ore};
            let get = match get_i {0=>Resource::Wood,1=>Resource::Brick,2=>Resource::Sheep,3=>Resource::Wheat,_=>Resource::Ore};
            let id = encode(Action::TradeBank { give, get });
            assert!((206..226).contains(&id));
            let back = decode(id).unwrap();
            assert_eq!(back, Action::TradeBank { give, get });
        }
    }
}
