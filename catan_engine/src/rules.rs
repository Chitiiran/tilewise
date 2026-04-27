//! Pure functions over GameState. No I/O, no globals.
//! Every rule is unit-testable by constructing a state and calling the function.

use crate::actions::Action;
use crate::events::GameEvent;
use crate::rng::Rng;
use crate::state::GameState;
use crate::state::GamePhase;

pub fn legal_actions(state: &GameState) -> Vec<Action> {
    // Implemented incrementally across Phases 4–5, dispatching on state.phase.
    // Each phase variant gets its own helper below.
    match &state.phase {
        crate::state::GamePhase::Setup1Place => legal_actions_setup_place(state),
        crate::state::GamePhase::Setup2Place => legal_actions_setup_place(state),
        crate::state::GamePhase::Roll => vec![],          // filled in Task 15
        crate::state::GamePhase::Main => vec![],          // filled in Task 16
        crate::state::GamePhase::Discard { .. } => vec![], // filled in Task 18
        crate::state::GamePhase::MoveRobber => vec![],     // filled in Task 19
        crate::state::GamePhase::Steal { .. } => vec![],   // filled in Task 19
        crate::state::GamePhase::Done { .. } => vec![],
    }
}

pub(crate) fn legal_actions_setup_place(state: &GameState) -> Vec<Action> {
    // Two sub-states:
    //   setup_pending == None  → place a settlement
    //   setup_pending == Some(v) → place a road adjacent to v
    match state.setup_pending {
        None => legal_setup_settlements(state),
        Some(v) => legal_setup_roads(state, v),
    }
}

fn legal_setup_settlements(state: &GameState) -> Vec<Action> {
    let mut out = Vec::with_capacity(54);
    for v in 0u8..54 {
        if is_legal_settlement_location(state, v) {
            out.push(Action::BuildSettlement(v));
        }
    }
    out
}

fn legal_setup_roads(state: &GameState, just_placed_vertex: u8) -> Vec<Action> {
    let mut out = Vec::new();
    for &e in &state.board.vertex_to_edges[just_placed_vertex as usize] {
        if state.roads[e as usize].is_none() {
            out.push(Action::BuildRoad(e));
        }
    }
    out
}

fn is_legal_settlement_location(state: &GameState, v: u8) -> bool {
    if state.settlements[v as usize].is_some() || state.cities[v as usize].is_some() {
        return false;
    }
    // Distance rule: no neighbor vertex may have a settlement or city.
    for &n in &state.board.vertex_to_vertices[v as usize] {
        if state.settlements[n as usize].is_some() || state.cities[n as usize].is_some() {
            return false;
        }
    }
    true
}

pub fn apply(state: &mut GameState, action: Action, rng: &mut Rng) -> Vec<GameEvent> {
    let mut events = Vec::new();
    match (&state.phase, action) {
        (GamePhase::Setup1Place, Action::BuildSettlement(v)) => {
            state.settlements[v as usize] = Some(state.current_player);
            state.vp[state.current_player as usize] += 1;
            state.setup_pending = Some(v);
            events.push(GameEvent::BuildSettlement { player: state.current_player, vertex: v });
        }
        (GamePhase::Setup1Place, Action::BuildRoad(e)) => {
            state.roads[e as usize] = Some(state.current_player);
            state.setup_pending = None;
            events.push(GameEvent::BuildRoad { player: state.current_player, edge: e });
            // Advance player; if all 4 placed, enter Setup-2 (reverse order, player 3 first).
            if state.current_player < 3 {
                state.current_player += 1;
            } else {
                state.phase = GamePhase::Setup2Place;
                // current_player stays at 3 — Setup-2 starts with player 3.
            }
        }
        _ => {
            // Other transitions implemented in Tasks 13–20.
            let _ = rng;
        }
    }
    events
}

pub fn is_terminal(state: &GameState) -> bool {
    state.is_terminal()
}
