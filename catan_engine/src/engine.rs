//! Top-level orchestrator.

use crate::actions::decode;
use crate::board::Board;
use crate::events::{EventLog, GameEvent};
use crate::rng::Rng;
use crate::rules::{apply, legal_actions};
use crate::state::GameState;
use std::sync::Arc;

pub struct Engine {
    pub state: GameState,
    pub rng: Rng,
    pub events: EventLog,
}

impl Engine {
    pub fn new(seed: u64) -> Self {
        let board = Arc::new(Board::standard());
        Self {
            state: GameState::new(board),
            rng: Rng::from_seed(seed),
            events: EventLog::new(),
        }
    }

    pub fn legal_actions(&self) -> Vec<u32> {
        legal_actions(&self.state)
            .into_iter()
            .map(crate::actions::encode)
            .collect()
    }

    pub fn step(&mut self, action_id: u32) {
        let action = decode(action_id).expect("invalid action ID");
        let evs = apply(&mut self.state, action, &mut self.rng);
        for e in evs {
            self.events.push(e);
        }
    }

    pub fn is_terminal(&self) -> bool {
        self.state.is_terminal()
    }

    pub fn event_log(&self) -> &[GameEvent] {
        self.events.as_slice()
    }
}
