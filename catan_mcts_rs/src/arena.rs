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

use crate::evaluator::TorchScriptEvaluator;
use crate::mcts::{best_action, Mcts};
use crate::mt19937::MtRng;
use crate::rng::NpRng;
use crate::state;
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
