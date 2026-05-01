//! Phase 2.1: port-aware trade ratios.

use catan_engine::actions::Action;
use catan_engine::board::{Board, Resource};
use catan_engine::rng::Rng;
use catan_engine::rules::{apply, legal_actions, trade_ratio_for, update_ports_owned_for_player};
use catan_engine::state::{
    GameState, GamePhase, PORT_BIT_GENERIC, PORT_BIT_WHEAT, PORT_BIT_BRICK,
};
use std::sync::Arc;

fn fresh() -> GameState {
    let mut state = GameState::new(Arc::new(Board::standard()));
    state.phase = GamePhase::Main;
    state.current_player = 0;
    state
}

#[test]
fn no_port_means_4_to_1_ratio() {
    let state = fresh();
    for give_idx in 0..5u8 {
        assert_eq!(trade_ratio_for(&state, 0, give_idx), 4);
    }
}

#[test]
fn generic_port_grants_3_to_1() {
    let mut state = fresh();
    state.ports_owned[0] = PORT_BIT_GENERIC;
    for give_idx in 0..5u8 {
        assert_eq!(trade_ratio_for(&state, 0, give_idx), 3);
    }
}

#[test]
fn specific_port_overrides_to_2_to_1_for_that_resource() {
    let mut state = fresh();
    state.ports_owned[0] = PORT_BIT_GENERIC | PORT_BIT_WHEAT;
    // Wheat → 2:1
    assert_eq!(trade_ratio_for(&state, 0, 3), 2);
    // Others → 3:1 (still have generic)
    assert_eq!(trade_ratio_for(&state, 0, 0), 3);
    assert_eq!(trade_ratio_for(&state, 0, 1), 3);
    assert_eq!(trade_ratio_for(&state, 0, 2), 3);
    assert_eq!(trade_ratio_for(&state, 0, 4), 3);
}

#[test]
fn specific_only_no_generic_other_resources_remain_4_to_1() {
    let mut state = fresh();
    state.ports_owned[0] = PORT_BIT_BRICK; // 2:1 brick port only
    assert_eq!(trade_ratio_for(&state, 0, 1), 2); // brick = 2:1
    // Other resources still 4:1.
    assert_eq!(trade_ratio_for(&state, 0, 0), 4);
    assert_eq!(trade_ratio_for(&state, 0, 3), 4);
}

#[test]
fn placing_settlement_on_port_vertex_grants_port() {
    let mut state = fresh();
    // Find a port vertex from the board.
    let port = state.board.ports[0]; // first port
    let v = port.vertices[0] as usize;
    // Settle there.
    state.settlements.set(v, Some(0));
    update_ports_owned_for_player(&mut state, 0);
    assert!(state.ports_owned[0] != 0, "should own at least one port");
}

#[test]
fn settlement_off_port_does_not_grant_port() {
    let mut state = fresh();
    // Find a vertex that's NOT in any port. We'll test some interior vertices.
    let port_vertices: std::collections::HashSet<u8> = state.board.ports.iter()
        .flat_map(|p| p.vertices.iter().copied())
        .collect();
    let off_port: u8 = (0..54u8).find(|v| !port_vertices.contains(v)).unwrap();
    state.settlements.set(off_port as usize, Some(0));
    update_ports_owned_for_player(&mut state, 0);
    assert_eq!(state.ports_owned[0], 0, "settlement off port grants nothing");
}

#[test]
fn legal_actions_uses_port_aware_ratio() {
    let mut state = fresh();
    state.hands[0] = [3, 0, 0, 0, 0]; // 3 wood
    state.bank[1] -= 3; // bank reflects player has 3 wood
    // Without port: 3 wood is < 4, no trade legal.
    let legal_no_port = legal_actions(&state);
    let trades_no_port: usize = legal_no_port.iter()
        .filter(|a| matches!(a, Action::TradeBank{..})).count();
    assert_eq!(trades_no_port, 0);

    // With generic 3:1 port: 3 wood is >= 3, 4 trade options (wood→{B,S,Wh,O}).
    state.ports_owned[0] = PORT_BIT_GENERIC;
    let legal_3to1 = legal_actions(&state);
    let trades_3to1: usize = legal_3to1.iter()
        .filter(|a| matches!(a, Action::TradeBank{..})).count();
    assert_eq!(trades_3to1, 4);
}

#[test]
fn apply_trade_respects_port_ratio() {
    let mut state = fresh();
    state.hands[0] = [3, 0, 0, 0, 0];
    state.bank[0] = 19 - 3; // accounting
    state.ports_owned[0] = PORT_BIT_GENERIC;
    let mut rng = Rng::from_seed(0);
    apply(&mut state, Action::TradeBank { give: Resource::Wood, get: Resource::Brick }, &mut rng);
    // 3 wood → 1 brick (3:1)
    assert_eq!(state.hands[0][0], 0, "all 3 wood given");
    assert_eq!(state.hands[0][1], 1);
    // Resource conservation.
    for r in 0..5 {
        let total: u32 = (0..4).map(|p| state.hands[p][r] as u32).sum::<u32>()
                            + state.bank[r] as u32;
        assert_eq!(total, 19);
    }
}
