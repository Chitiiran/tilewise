//! Top-level orchestrator.

use crate::actions::decode;
use crate::board::Board;
use crate::events::{EventLog, GameEvent};
use crate::rng::Rng;
use crate::rules::{apply, legal_actions};
use crate::state::GameState;
use crate::stats::GameStats;
use std::sync::Arc;

pub struct Engine {
    pub state: GameState,
    pub rng: Rng,
    pub events: EventLog,
    pub stats: GameStats,
}

impl Engine {
    pub fn new(seed: u64) -> Self {
        let board = Arc::new(Board::standard());
        Self {
            state: GameState::new(board),
            rng: Rng::from_seed(seed),
            events: EventLog::new(),
            stats: GameStats::new(),
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
        for e in &evs {
            self.stats.fold_event(e);
            self.events.push(*e);
        }
        // Update cards_in_hand_max
        for p in 0..4 {
            let total: u32 = self.state.hands[p].iter().map(|&x| x as u32).sum();
            if total > self.stats.players[p].cards_in_hand_max {
                self.stats.players[p].cards_in_hand_max = total;
            }
        }
        // Lock vp_final into stats once the game ends
        if self.state.is_terminal() {
            for p in 0..4 {
                self.stats.players[p].vp_final = self.state.vp[p];
            }
        }
    }

    pub fn is_terminal(&self) -> bool {
        self.state.is_terminal()
    }

    /// Returns true when the next action must come from the environment (dice / steal),
    /// not from a player. Used by OpenSpiel chance-node modeling.
    pub fn is_chance_pending(&self) -> bool {
        matches!(self.state.phase, crate::state::GamePhase::Roll)
            || matches!(self.state.phase, crate::state::GamePhase::Steal { .. })
    }

    pub fn event_log(&self) -> &[GameEvent] {
        self.events.as_slice()
    }

    pub fn stats(&self) -> &GameStats {
        &self.stats
    }
}
