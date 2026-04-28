//! Top-level orchestrator.

use crate::actions::decode;
use crate::board::Board;
use crate::events::{EventLog, GameEvent};
use crate::rng::Rng;
use crate::rules::{apply, legal_actions};
use crate::state::GameState;
use crate::stats::GameStats;
use std::sync::Arc;

#[derive(Clone)]
pub struct Engine {
    pub state: GameState,
    pub rng: Rng,
    pub events: EventLog,
    pub stats: GameStats,
    pub history: Vec<u32>,
}

impl Engine {
    pub fn new(seed: u64) -> Self {
        let board = Arc::new(Board::standard());
        Self {
            state: GameState::new(board),
            rng: Rng::from_seed(seed),
            events: EventLog::new(),
            stats: GameStats::new(),
            history: Vec::new(),
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
        self.history.push(action_id);
        let evs = apply(&mut self.state, action, &mut self.rng);
        self.record_events(&evs);
    }

    /// Single sink for stats + event-log updates after a state mutation.
    /// Both `step` and `apply_chance_outcome` route through here so the bookkeeping
    /// (cards_in_hand_max, vp_final lock-in on terminal) cannot drift between paths.
    fn record_events(&mut self, evs: &[GameEvent]) {
        for e in evs {
            self.stats.fold_event(e);
            self.events.push(*e);
        }
        for p in 0..4 {
            let total: u32 = self.state.hands[p].iter().map(|&x| x as u32).sum();
            if total > self.stats.players[p].cards_in_hand_max {
                self.stats.players[p].cards_in_hand_max = total;
            }
        }
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

    /// Discrete chance outcomes at the current state. Each entry is `(outcome_value, probability)`.
    ///
    /// Outcome semantics:
    ///   - In `Roll` phase: outcome_value is the dice sum, 2..=12.
    ///   - In `Steal` phase: outcome_value packs `(victim_index_in_from_options, card_index_in_victim_hand)`
    ///     as `victim * 256 + card_index`. The card_index is a flat index 0..total_cards into the victim's
    ///     hand: 0..hand[wood], hand[wood]..hand[wood]+hand[brick], etc. This matches the offset arithmetic
    ///     in `apply_chance_outcome`.
    ///
    /// Panics if not at a chance node.
    pub fn chance_outcomes(&self) -> Vec<(u32, f64)> {
        use crate::state::GamePhase;
        match &self.state.phase {
            GamePhase::Roll => {
                // 2d6 distribution: counts of (d1+d2) for d1,d2 in 1..=6.
                let mut counts = [0u32; 13]; // index = sum
                for d1 in 1..=6 {
                    for d2 in 1..=6 {
                        counts[d1 + d2] += 1;
                    }
                }
                let mut out = Vec::with_capacity(11);
                for s in 2..=12u32 {
                    out.push((s, counts[s as usize] as f64 / 36.0));
                }
                out
            }
            GamePhase::Steal { from_options } => {
                // For Tier 1 we steal from the FIRST victim in from_options (matches old behavior).
                // Each card in the victim's hand is an equally likely outcome.
                let victim = from_options[0];
                let hand = &self.state.hands[victim as usize];
                let total: u32 = hand.iter().map(|&x| x as u32).sum();
                assert!(total > 0, "Steal phase entered with empty victim hand");
                let mut out = Vec::with_capacity(total as usize);
                let p = 1.0 / total as f64;
                for card_index in 0..total {
                    out.push(((victim as u32) * 256 + card_index, p));
                }
                out
            }
            _ => panic!("chance_outcomes() called outside a chance node"),
        }
    }

    /// Drive a chance node by supplying the outcome value (one of those returned by `chance_outcomes()`).
    /// Records the resulting events in the event log just like `step()` does.
    /// Panics if not at a chance node, or if `value` is not a valid outcome for the current phase.
    pub fn apply_chance_outcome(&mut self, value: u32) {
        use crate::state::GamePhase;
        let evs = match self.state.phase.clone() {
            GamePhase::Roll => {
                let roll = u8::try_from(value).expect("dice roll out of u8 range");
                assert!((2..=12).contains(&roll), "invalid dice sum {roll}");
                self.history.push(0x8000_0000 | value);
                crate::rules::apply_dice_roll(&mut self.state, roll)
            }
            GamePhase::Steal { from_options } => {
                let victim = (value / 256) as u8;
                let card_index = (value % 256) as u8;
                assert_eq!(
                    victim, from_options[0],
                    "value's victim {} != from_options[0] {} — Tier 1 steals only from first option",
                    victim, from_options[0],
                );
                self.history.push(0x8000_0000 | value);
                crate::rules::apply_steal(&mut self.state, victim, card_index)
            }
            _ => panic!("apply_chance_outcome() called outside a chance node"),
        };
        self.record_events(&evs);
    }

    pub fn event_log(&self) -> &[GameEvent] {
        self.events.as_slice()
    }

    pub fn stats(&self) -> &GameStats {
        &self.stats
    }

    pub fn action_history(&self) -> &[u32] {
        &self.history
    }
}
