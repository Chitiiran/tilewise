//! Top-level orchestrator.

use crate::actions::decode;
use crate::board::Board;
use crate::events::{EventLog, GameEvent};
use crate::rng::Rng;
use crate::rules::{apply, legal_actions};
use crate::state::GameState;
use crate::stats::GameStats;
use std::sync::Arc;

/// Maximum number of state mutations a single `random_rollout_to_terminal` call
/// may perform before bailing with `[0.0; 4]` (no winner). Empirically Tier-1
/// games under uniformly-random play take 4-12k steps; 30k leaves headroom for
/// the long-tail distribution while bounding the worst case sharply enough to
/// prevent multi-hour single-game stalls in MCTS rollouts.
///
/// v2 (2026-04-28): lowered from 100_000 to 30_000 after observing that the
/// 100k cap allowed individual MCTS games at sims=100 to stall for 5+ hours
/// when entering a state region with many cap-firing rollouts.
pub const ROLLOUT_STEP_CAP: u32 = 30_000;

#[derive(Clone)]
pub struct Engine {
    pub state: GameState,
    pub rng: Rng,
    pub events: EventLog,
    pub stats: GameStats,
    pub history: Vec<u32>,
}

impl Engine {
    /// Construct an engine with the **balanced (ABC) board layout** seeded by `seed`.
    /// Phase 2.6: ABC is the v2 default. Use `Engine::with_standard_board(seed)` for
    /// the canonical fixed layout (only used by older tests).
    pub fn new(seed: u64) -> Self {
        let board = Arc::new(Board::generate_abc(seed));
        Self {
            state: GameState::new(board),
            rng: Rng::from_seed(seed),
            events: EventLog::new(),
            stats: GameStats::new(),
            history: Vec::new(),
        }
    }

    /// Construct an engine with the **fixed canonical board** (Catan beginner setup).
    /// Useful for v1 regression testing or strict reproducibility against v1 hashes.
    pub fn with_standard_board(seed: u64) -> Self {
        let board = Arc::new(Board::standard());
        Self {
            state: GameState::new(board),
            rng: Rng::from_seed(seed),
            events: EventLog::new(),
            stats: GameStats::new(),
            history: Vec::new(),
        }
    }

    pub fn legal_actions(&mut self) -> Vec<u32> {
        // Cached path: when the legal_mask cache is fresh, materialize the Vec
        // from set bits — much faster than re-running legal_actions() through
        // the rules layer (which re-iterates every vertex/edge/dev-card
        // branch). MCTS hits this path heavily because it queries the same
        // state many times between mutations.
        if !self.state.legal_mask_dirty {
            return self.state.legal_mask_cached.iter_ids().collect();
        }
        // Cache miss: recompute via rules. Don't populate the cache here —
        // pure-bench profiles show the cache-populate (LegalMask::from_action_ids
        // is ~600 ns) costs more than it saves for single-pass callers (random
        // self-play). Callers that want the cache populated (MCTS) call
        // legal_mask() instead, which has its own populate path.
        legal_actions(&self.state)
            .into_iter()
            .map(crate::actions::encode)
            .collect()
    }

    /// Legal-action bitmap. Bit i set ⇔ action i is legal in the current state.
    /// See `actions::LegalMask`. Phase 2.5: caches the result, recomputes only
    /// when state mutations marked the cache dirty. ~50× speedup when the same
    /// state is queried multiple times (typical MCTS pattern).
    pub fn legal_mask(&mut self) -> crate::actions::LegalMask {
        if self.state.legal_mask_dirty {
            let ids: Vec<u32> = legal_actions(&self.state)
                .into_iter()
                .map(crate::actions::encode)
                .collect();
            self.state.legal_mask_cached = crate::actions::LegalMask::from_action_ids(&ids);
            self.state.legal_mask_dirty = false;
        }
        self.state.legal_mask_cached
    }

    pub fn step(&mut self, action_id: u32) {
        let action = decode(action_id).expect("invalid action ID");
        self.history.push(action_id);
        let evs = apply(&mut self.state, action, &mut self.rng);
        self.state.legal_mask_dirty = true;
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
                crate::rules::apply_dice_roll(&mut self.state, roll, &mut self.rng)
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
        self.state.legal_mask_dirty = true;
        self.record_events(&evs);
    }

    /// Run a full random self-play rollout from the current state until terminal.
    /// Returns AlphaZero-style terminal rewards: +1 for the winner, -1 for losers,
    /// or [0;4] if the rollout hit the safety cap before terminating.
    ///
    /// Uses a fresh SmallRng seeded by `rollout_seed` for chance-node sampling and
    /// uniform legal-action selection. Does NOT consume self.rng — so the same
    /// engine state can be used as the root of multiple independent rollouts via
    /// `engine.clone().random_rollout_to_terminal(seed_i)`.
    ///
    /// The engine ends in a terminal (or step-capped) state. Caller should clone
    /// first if they need to preserve the pre-rollout state.
    pub fn random_rollout_to_terminal(&mut self, rollout_seed: u64) -> [f32; 4] {
        use rand::{Rng as _, SeedableRng};
        use rand::rngs::SmallRng;
        let mut rng = SmallRng::seed_from_u64(rollout_seed);
        let mut steps = 0u32;

        while !self.is_terminal() && steps < ROLLOUT_STEP_CAP {
            if self.is_chance_pending() {
                // Sample a chance outcome by probability.
                let outcomes = self.chance_outcomes();
                let r: f64 = rng.gen();
                let mut cum = 0.0f64;
                let mut chosen = outcomes.last().unwrap().0;
                for (v, p) in &outcomes {
                    cum += p;
                    if r <= cum {
                        chosen = *v;
                        break;
                    }
                }
                self.apply_chance_outcome(chosen);
            } else {
                let legal = self.legal_actions();
                if legal.is_empty() {
                    break;
                }
                let idx = rng.gen_range(0..legal.len());
                self.step(legal[idx]);
            }
            steps += 1;
        }

        if let crate::state::GamePhase::Done { winner } = self.state.phase {
            let mut returns = [-1.0f32; 4];
            returns[winner as usize] = 1.0;
            returns
        } else {
            // Capped or stuck — no winner.
            [0.0f32; 4]
        }
    }

    /// Depth-bounded greedy evaluator. Returns a value vector in [-1, 1] suitable
    /// for use as an OpenSpiel MCTS `Evaluator.evaluate` result.
    ///
    /// Semantics:
    ///   - Plays forward up to `depth * 4` EndTurn-equivalents (i.e. roughly
    ///     `depth` rounds of the game), or until terminal, whichever comes first.
    ///   - Decision actions: each player picks the highest-priority legal action
    ///     (ties broken by lowest action_id; matches Python `_action_priority`).
    ///   - Chance actions: sampled uniformly by probability using a SmallRng
    ///     seeded by `eval_seed`.
    ///   - Terminal: returns AlphaZero-style returns (+1 winner, -1 losers).
    ///   - Non-terminal: returns per-player VP scaled to [-1, 1] where
    ///     `normalize(vp) = (vp / 10.0) * 2.0 - 1.0`. This matches the
    ///     winning-VP threshold of 10, so a player at 10 VP -> +1.0.
    ///
    /// Mutates self into the post-lookahead state — caller should clone first
    /// if they need to preserve the pre-lookahead engine.
    pub fn lookahead_vp_value(&mut self, depth: u32, eval_seed: u64) -> [f32; 4] {
        use rand::{Rng as _, SeedableRng};
        use rand::rngs::SmallRng;

        // Already terminal? Return AlphaZero returns directly.
        if let crate::state::GamePhase::Done { winner } = self.state.phase {
            let mut returns = [-1.0f32; 4];
            returns[winner as usize] = 1.0;
            return returns;
        }

        // depth=0 means evaluate at the current state, no forward play.
        if depth == 0 {
            return Self::vp_to_value(&self.state.vp);
        }

        let mut rng = SmallRng::seed_from_u64(eval_seed);
        let mut end_turns_remaining = depth * 4;
        // Bound steps to prevent any pathological non-terminating subtree under
        // greedy play. depth*4 EndTurns each take O(20) decision steps + chance,
        // so depth=25 -> ~2k steps; 50k cap is generous safety margin.
        let mut steps = 0u32;
        let step_cap = (depth * 4 * 200).max(2000);

        while !self.is_terminal() && end_turns_remaining > 0 && steps < step_cap {
            if self.is_chance_pending() {
                let outcomes = self.chance_outcomes();
                let r: f64 = rng.gen();
                let mut cum = 0.0f64;
                let mut chosen = outcomes.last().unwrap().0;
                for (v, p) in &outcomes {
                    cum += p;
                    if r <= cum {
                        chosen = *v;
                        break;
                    }
                }
                self.apply_chance_outcome(chosen);
            } else {
                let legal = self.legal_actions();
                if legal.is_empty() {
                    break;
                }
                let action = Self::greedy_pick(&legal);
                if action == 204 {
                    end_turns_remaining = end_turns_remaining.saturating_sub(1);
                }
                self.step(action);
            }
            steps += 1;
        }

        // Terminal during lookahead -> AlphaZero returns.
        if let crate::state::GamePhase::Done { winner } = self.state.phase {
            let mut returns = [-1.0f32; 4];
            returns[winner as usize] = 1.0;
            return returns;
        }
        Self::vp_to_value(&self.state.vp)
    }

    /// Mirrors Python `_action_priority` in `mcts_study/catan_mcts/bots.py`.
    /// Higher score = bot prefers it. Ties broken by lowest action_id (deterministic).
    fn action_priority(action_id: u32) -> i32 {
        match action_id {
            54..=107  => 100, // city — best
            0..=53    => 80,  // settlement
            108..=179 => 50,  // road
            205       => 10,  // roll dice
            204       => 1,   // end turn
            180..=198 => 5,   // robber move
            199..=203 => 5,   // discard
            _ => 0,
        }
    }

    fn greedy_pick(legal: &[u32]) -> u32 {
        let mut best = legal[0];
        let mut best_score = Self::action_priority(best);
        for &a in &legal[1..] {
            let s = Self::action_priority(a);
            if s > best_score || (s == best_score && a < best) {
                best = a;
                best_score = s;
            }
        }
        best
    }

    fn vp_to_value(vp: &[u8; 4]) -> [f32; 4] {
        let mut out = [0f32; 4];
        for p in 0..4 {
            // 0 VP -> -1.0, 10 VP -> +1.0; clamp to [-1, 1] for safety.
            let v = (vp[p] as f32 / 10.0) * 2.0 - 1.0;
            out[p] = v.clamp(-1.0, 1.0);
        }
        out
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
