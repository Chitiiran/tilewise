//! Regression test for the v2 stuck-game pathology (seed 725001):
//!
//! Before the cap, MCTS could loop ProposeTrade(A→B), ProposeTrade(B→A),
//! ProposeTrade(A→B), ... forever within a single turn — total resources
//! conserved, no progress. Cap at MAX_TRADES_PER_TURN = 4.

use catan_engine::actions::Action;
use catan_engine::board::Resource;
use catan_engine::state::{GamePhase, MAX_TRADES_PER_TURN};
use catan_engine::Engine;

/// Drive the engine into Main phase for player 0 with both Wood and Brick in hand
/// (so ProposeTrade(Wood→Brick) and ProposeTrade(Brick→Wood) are legal).
fn into_main_with_resources() -> Engine {
    let mut e = Engine::new(42);
    // Cheat: directly set hands. Tests touch state internals legally.
    // Skip past Setup by stepping through whatever legal action exists; for
    // this targeted test we shortcut by forcing state.
    e.state.phase = GamePhase::Main;
    e.state.current_player = 0;
    // Give player 0 some of each resource so trades are legal regardless of
    // which direction MCTS picks. Give every opponent both wood and brick so
    // the legality "any opponent has the get resource" check always passes.
    for r in 0..5 {
        e.state.hands[0][r] = 5;
        for p in 1..4 {
            e.state.hands[p][r] = 3;
        }
    }
    e.state.legal_mask_dirty = true;
    e
}

fn count_propose_trade_actions(legal: &[Action]) -> usize {
    legal.iter().filter(|a| matches!(a, Action::ProposeTrade { .. })).count()
}

#[test]
fn trades_this_turn_starts_at_zero() {
    let e = into_main_with_resources();
    assert_eq!(e.state.trades_this_turn, 0);
}

#[test]
fn propose_trade_increments_counter() {
    let mut e = into_main_with_resources();
    let action = Action::ProposeTrade { give: Resource::Wood, get: Resource::Brick };
    let id = catan_engine::actions::encode(action);
    e.step(id);
    assert_eq!(e.state.trades_this_turn, 1);
}

#[test]
fn propose_trade_disallowed_after_cap() {
    let mut e = into_main_with_resources();
    let id = catan_engine::actions::encode(
        Action::ProposeTrade { give: Resource::Wood, get: Resource::Brick }
    );
    // Burn through MAX_TRADES_PER_TURN trades — they alternate direction so
    // hands stay non-empty.
    let alt = catan_engine::actions::encode(
        Action::ProposeTrade { give: Resource::Brick, get: Resource::Wood }
    );
    for i in 0..MAX_TRADES_PER_TURN {
        let next = if i % 2 == 0 { id } else { alt };
        e.step(next);
    }
    assert_eq!(e.state.trades_this_turn, MAX_TRADES_PER_TURN);

    // Now legal_actions must NOT contain any ProposeTrade.
    let legal = catan_engine::rules::legal_actions(&e.state);
    assert_eq!(
        count_propose_trade_actions(&legal), 0,
        "after {} trades, ProposeTrade must be disallowed; got {} legal trade actions",
        MAX_TRADES_PER_TURN,
        count_propose_trade_actions(&legal),
    );
}

#[test]
fn end_turn_resets_trade_counter() {
    let mut e = into_main_with_resources();
    let id = catan_engine::actions::encode(
        Action::ProposeTrade { give: Resource::Wood, get: Resource::Brick }
    );
    e.step(id);
    assert_eq!(e.state.trades_this_turn, 1);
    let end_turn = catan_engine::actions::encode(Action::EndTurn);
    e.step(end_turn);
    assert_eq!(e.state.trades_this_turn, 0,
        "trades_this_turn must reset on EndTurn");
}

#[test]
fn cap_constant_in_sane_range() {
    // Sanity: cap must be >= 1 (otherwise trades are entirely disabled, which
    // changes legal_actions semantics) and small enough to prevent the
    // infinite-loop pathology we saw at unbounded.
    // Phase 3.4 data showed ProposeTrade was 47% of all actions even at cap=4;
    // dropping to 1 trims pure trade-loop noise without losing the
    // "trade then build" pattern.
    assert!(MAX_TRADES_PER_TURN >= 1);
    assert!(MAX_TRADES_PER_TURN <= 8);
}
