//! Pure functions over GameState. No I/O, no globals.
//! Every rule is unit-testable by constructing a state and calling the function.

use crate::actions::Action;
use crate::events::GameEvent;
use crate::rng::Rng;
use crate::state::GameState;

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

pub(crate) fn legal_actions_setup_place(_state: &GameState) -> Vec<Action> {
    // Filled in Task 12.
    vec![]
}

pub fn apply(state: &mut GameState, action: Action, rng: &mut Rng) -> Vec<GameEvent> {
    // Implemented incrementally. Stub for now.
    let _ = (state, action, rng);
    vec![]
}

pub fn is_terminal(state: &GameState) -> bool {
    state.is_terminal()
}
