//! Arena game driver — bit-exact port of catan_az.arena._play_arena_game +
//! seating/seed_plan/tiebreak.
//!
//! DUAL RNG (matches the Python oracle exactly):
//!   - GAME-level chance fast-path uses CPython random.Random(seed) (MtRng).
//!   - each seat's MCTS uses its own NpRng: cand = default_rng(seed+11),
//!     champ = default_rng(seed+13). Arena is greedy (eps=0), so the MCTS RNG
//!     only drives MCTS-internal chance sampling.
//!
//! Returns (winner_seat or -1, timed_out, vp_margin) per game, plus the seat
//! roles, so Python aggregates the existing ArenaResult unchanged.

use crate::evaluator::{NetOutput, TorchScriptEvaluator};
use crate::mcts::{best_action, Mcts, SearchSession, SessionStep};
use crate::mt19937::MtRng;
use crate::rng::NpRng;
use crate::state;
use catan_engine::observation::Observation;
use catan_engine::Engine;

const C_PUCT: f64 = 1.4;

/// _BASE = ["cand","champ","cand","champ"]; rotation = BASE[rot:]+BASE[:rot].
/// Returns true where the seat is the candidate.
pub fn seating_is_cand(rot: usize) -> [bool; 4] {
    let base = [true, false, true, false]; // cand, champ, cand, champ
    let mut out = [false; 4];
    for i in 0..4 {
        out[i] = base[(i + rot) % 4];
    }
    out
}

/// seed_plan: per_rot = games/4; (rot, seed) with seed = base + rot*10000 + i.
pub fn seed_plan(seed_base: u64, games: usize) -> Vec<(usize, u64)> {
    let per_rot = games / 4;
    let mut plan = Vec::new();
    for rot in 0..4 {
        for i in 0..per_rot {
            plan.push((rot, seed_base + (rot as u64) * 10_000 + i as u64));
        }
    }
    plan
}

fn new_engine(seed: u64, vp_target: u8, bonuses: bool) -> Engine {
    if vp_target == 10 && bonuses {
        Engine::new(seed)
    } else {
        Engine::with_rules(seed, vp_target, bonuses)
    }
}

pub struct ArenaGameResult {
    pub winner_seat: i32,
    pub timed_out: bool,
    pub vp_margin: i32,
}

/// Play one arena game. `ev_cand`/`ev_champ` are the two nets. `seating_cand[s]`
/// tells whether seat s is the candidate. Greedy (no Dirichlet/temperature).
/// max_steps mirrors the Python 200_000 cap (no wall-clock deadline here — the
/// Rust path finishes games naturally; the whole point of the rewrite).
#[allow(clippy::too_many_arguments)]
pub fn play_arena_game(
    ev_cand: &TorchScriptEvaluator,
    ev_champ: &TorchScriptEvaluator,
    seed: u64,
    seating_cand: [bool; 4],
    sims: u32,
    vp_target: u8,
    bonuses: bool,
    max_steps: u32,
) -> ArenaGameResult {
    let mut engine = new_engine(seed, vp_target, bonuses);
    let mut chance_rng = MtRng::from_seed(seed); // stdlib random.Random(seed)
    // Per-seat greedy MCTS RNGs (seed+11 cand, seed+13 champ).
    let mut rng_cand = NpRng::from_seed(seed.wrapping_add(11));
    let mut rng_champ = NpRng::from_seed(seed.wrapping_add(13));
    let mut mcts_cand = Mcts::new(ev_cand, C_PUCT, 0.8, 0.0);
    let mut mcts_champ = Mcts::new(ev_champ, C_PUCT, 0.8, 0.0);

    let mut steps = 0u32;
    while !engine.is_terminal() && steps < max_steps {
        if engine.is_chance_pending() {
            // GAME-level chance via the stdlib MT rng (cumulative search).
            let outs = engine.chance_outcomes();
            let r = chance_rng.random_f64();
            let mut cum = 0.0f64;
            let mut chosen = outs.last().unwrap().0;
            for &(v, p) in &outs {
                cum += p;
                if r <= cum {
                    chosen = v;
                    break;
                }
            }
            engine.apply_chance_outcome(chosen);
            steps += 1;
            continue;
        }
        let legal = engine.legal_actions();
        if legal.len() == 1 {
            engine.step(legal[0]);
            steps += 1;
            continue;
        }
        let cp = engine.state.current_player as usize;
        let action = if seating_cand[cp] {
            let vc = mcts_cand.search(engine.clone(), sims, &mut rng_cand);
            best_action(&vc) as u32
        } else {
            let vc = mcts_champ.search(engine.clone(), sims, &mut rng_champ);
            best_action(&vc) as u32
        };
        engine.step(action);
        steps += 1;
    }

    if !engine.is_terminal() {
        // step-cap exit -> VP tiebreak, marked timed_out.
        return ArenaGameResult {
            winner_seat: state::vp_leader_margin(&engine),
            timed_out: true,
            vp_margin: state::vp_margin(&engine),
        };
    }
    // Natural finish: winner = returns index of +1.0, else -1.
    let rets = state::returns_abs(&engine);
    let winner = rets.iter().position(|&r| r == 1.0).map(|i| i as i32).unwrap_or(-1);
    ArenaGameResult {
        winner_seat: winner,
        timed_out: false,
        vp_margin: state::vp_margin(&engine),
    }
}

// ---------------------------------------------------------------------------
// Task 1: pausable per-game arena state (Task 10's Slot pattern, ported to the
// arena's dual-RNG contract). Drives the SAME SearchSession algorithm as
// play_arena_game but YIELDS at each leaf net-eval so Task 2's scheduler can
// batch leaves across many concurrent arena games into one evaluate_batch
// call. See selfplay.rs's Slot for the donor pattern; the differences are:
// game-level chance uses MtRng (byte-copy of the walk above, not
// state::sample_chance), TWO per-seat MCTS RNGs (cand/champ) selected by
// seating_cand[current_player], greedy search (no Dirichlet/temperature), and
// no move recording — only the final ArenaGameResult.
// ---------------------------------------------------------------------------

pub struct ArenaSlot {
    pub engine: Engine,
    pub chance_rng: MtRng, // game-level chance (arena contract)
    pub rng_cand: NpRng,   // seed+11 — MCTS chance for cand searches
    pub rng_champ: NpRng,  // seed+13 — champ searches
    pub seating_cand: [bool; 4],
    pub seed: u64,
    pub rot: usize,
    pub session: Option<SearchSession>,
    pub cur_is_cand: bool, // net owning the CURRENT session
    pub steps: u32,
    pub done: bool,
    pub result: Option<ArenaGameResult>,
}

impl ArenaSlot {
    pub fn new(rot: usize, seed: u64, vp_target: u8, bonuses: bool) -> Self {
        ArenaSlot {
            engine: new_engine(seed, vp_target, bonuses),
            chance_rng: MtRng::from_seed(seed),
            rng_cand: NpRng::from_seed(seed.wrapping_add(11)),
            rng_champ: NpRng::from_seed(seed.wrapping_add(13)),
            seating_cand: seating_is_cand(rot),
            seed,
            rot,
            session: None,
            cur_is_cand: false,
            steps: 0,
            done: false,
            result: None,
        }
    }

    /// The rng matching cur_is_cand (for provide()).
    pub fn cur_rng(&mut self) -> &mut NpRng {
        if self.cur_is_cand {
            &mut self.rng_cand
        } else {
            &mut self.rng_champ
        }
    }

    /// Feed the net output for the current session's parked leaf, routing to
    /// the correct per-seat rng internally (split borrows of struct fields
    /// avoid the session+rng aliasing conflict callers would hit doing this
    /// by hand).
    pub fn provide_cur(&mut self, out: &NetOutput) -> SessionStep {
        let rng = if self.cur_is_cand { &mut self.rng_cand } else { &mut self.rng_champ };
        self.session.as_mut().expect("provide_cur without session").provide(out, rng)
    }

    /// Advance through chance (MtRng cumulative walk, byte-copy of
    /// arena.rs:86-97) and single-legal fast-paths; at a decision node start
    /// a SearchSession (greedy: dirichlet_eps=0.0, alpha=0.8, c=1.4) and pump
    /// it with the MOVER's rng; set cur_is_cand. Returns the parked leaf obs,
    /// or None when the game reached terminal/max_steps (sets done + result
    /// exactly like arena.rs:119-134).
    pub fn advance_to_search(&mut self, sims: u32, max_steps: u32) -> Option<Observation> {
        debug_assert!(self.session.is_none(), "advance_to_search called with a parked session");
        loop {
            if self.engine.is_terminal() || self.steps >= max_steps {
                self.finish_game();
                return None;
            }
            if self.engine.is_chance_pending() {
                let outs = self.engine.chance_outcomes();
                let r = self.chance_rng.random_f64();
                let mut cum = 0.0f64;
                let mut chosen = outs.last().unwrap().0;
                for &(v, p) in &outs {
                    cum += p;
                    if r <= cum {
                        chosen = v;
                        break;
                    }
                }
                self.engine.apply_chance_outcome(chosen);
                self.steps += 1;
                continue;
            }
            let legal = self.engine.legal_actions();
            if legal.len() == 1 {
                self.engine.step(legal[0]);
                self.steps += 1;
                continue;
            }
            let cp = self.engine.state.current_player as usize;
            self.cur_is_cand = self.seating_cand[cp];
            let mut sess = SearchSession::new(self.engine.clone(), sims, C_PUCT, 0.8, 0.0);
            let rng = if self.cur_is_cand { &mut self.rng_cand } else { &mut self.rng_champ };
            match sess.pump(rng) {
                SessionStep::NeedEval(obs) => {
                    self.session = Some(sess);
                    return Some(obs);
                }
                SessionStep::Done => {
                    self.session = Some(sess);
                    self.finish_move();
                    continue;
                }
            }
        }
    }

    /// Session Done: take_visit_counts -> best_action -> engine.apply; clears
    /// session, bumps steps.
    pub fn finish_move(&mut self) {
        let sess = self.session.take().expect("finish_move without session");
        let visit_counts = sess.take_visit_counts();
        let action = best_action(&visit_counts) as u32;
        self.engine.step(action);
        self.steps += 1;
    }

    fn finish_game(&mut self) {
        self.done = true;
        self.result = Some(if !self.engine.is_terminal() {
            ArenaGameResult {
                winner_seat: state::vp_leader_margin(&self.engine),
                timed_out: true,
                vp_margin: state::vp_margin(&self.engine),
            }
        } else {
            let rets = state::returns_abs(&self.engine);
            let winner = rets.iter().position(|&r| r == 1.0).map(|i| i as i32).unwrap_or(-1);
            ArenaGameResult { winner_seat: winner, timed_out: false, vp_margin: state::vp_margin(&self.engine) }
        });
    }
}

/// Task 2: cross-game leaf batching for the arena, over TWO nets. Each active
/// game's parked leaf belongs to whichever net owns the CURRENT session
/// (`ArenaSlot::cur_is_cand`); the scheduler partitions active games into a
/// cand-queue and a champ-queue each iteration (ownership can flip between a
/// slot's sessions, so it is recomputed every pass, not cached) and flushes
/// each queue in `b_max`-sized chunks against its own evaluator. Mirrors
/// `selfplay::play_games_batched` (selfplay.rs:298-335); returns results in
/// `pairs` order regardless of which game finishes first.
#[allow(clippy::too_many_arguments)]
pub fn play_arena_games_batched(
    ev_cand: &TorchScriptEvaluator,
    ev_champ: &TorchScriptEvaluator,
    pairs: &[(usize, u64)],
    sims: u32,
    vp_target: u8,
    bonuses: bool,
    max_steps: u32,
) -> Vec<ArenaGameResult> {
    let b_max_cand =
        ev_cand.b_max().expect("play_arena_games_batched needs a batched cand evaluator");
    let b_max_champ =
        ev_champ.b_max().expect("play_arena_games_batched needs a batched champ evaluator");

    let mut slots: Vec<ArenaSlot> =
        pairs.iter().map(|&(rot, seed)| ArenaSlot::new(rot, seed, vp_target, bonuses)).collect();
    let mut parked: Vec<Option<Observation>> =
        slots.iter_mut().map(|sl| sl.advance_to_search(sims, max_steps)).collect();

    loop {
        let active: Vec<usize> = (0..slots.len()).filter(|&i| parked[i].is_some()).collect();
        if active.is_empty() {
            break;
        }
        let (q_cand, q_champ): (Vec<usize>, Vec<usize>) =
            active.into_iter().partition(|&i| slots[i].cur_is_cand);

        for (queue, ev, b_max) in
            [(q_cand, ev_cand, b_max_cand), (q_champ, ev_champ, b_max_champ)]
        {
            for chunk in queue.chunks(b_max) {
                let obs_refs: Vec<&Observation> =
                    chunk.iter().map(|&i| parked[i].as_ref().unwrap()).collect();
                let outs = ev.evaluate_batch(&obs_refs);
                for (pos, &i) in chunk.iter().enumerate() {
                    match slots[i].provide_cur(&outs[pos]) {
                        SessionStep::NeedEval(obs) => parked[i] = Some(obs),
                        SessionStep::Done => {
                            slots[i].finish_move();
                            parked[i] = slots[i].advance_to_search(sims, max_steps);
                        }
                    }
                }
            }
        }
    }

    slots.into_iter().map(|s| s.result.expect("slot finished without result")).collect()
}
