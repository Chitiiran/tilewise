//! Thin state helpers over `catan_engine::Engine` for the MCTS/self-play/arena
//! layers. The engine itself is already Rust; this module only adds the
//! adapter-level semantics that `catan_mcts.adapter.CatanState` provides on the
//! Python side (the oracle), so the Rust path matches it exactly:
//!   - current_player mapping (terminal / chance sentinels)
//!   - returns(): absolute-seat AlphaZero returns (+1 winner, -1 losers, 0 none)
//!   - chance-node sampling via the numpy-replica RNG (cumulative search)
//!   - vp / vp_margin / vp_leader (arena tiebreak)
//!
//! Player-id sentinels mirror pyspiel: TERMINAL = -4, CHANCE = -1. Only the
//! ordering relative to real seats (>=0) matters for MCTS, which checks
//! `to_play >= 0`.

use crate::rng::NpRng;
use catan_engine::Engine;

pub const TERMINAL_PLAYER: i32 = -4;
pub const CHANCE_PLAYER: i32 = -1;
pub const N_PLAYERS: usize = 4;

/// pyspiel-style current_player(): TERMINAL/CHANCE sentinels, else the seat.
pub fn current_player(engine: &Engine) -> i32 {
    if engine.is_terminal() {
        TERMINAL_PLAYER
    } else if engine.is_chance_pending() {
        CHANCE_PLAYER
    } else {
        engine.state.current_player as i32
    }
}

/// Absolute-seat AlphaZero returns, matching CatanState.returns():
/// +1 for the winner, -1 for the others, all 0 if there is no winner.
pub fn returns_abs(engine: &Engine) -> [f32; N_PLAYERS] {
    if !engine.is_terminal() {
        return [0.0; N_PLAYERS];
    }
    let winner = engine.stats().winner_player_id;
    if winner < 0 {
        return [0.0; N_PLAYERS];
    }
    let mut out = [-1.0f32; N_PLAYERS];
    out[winner as usize] = 1.0;
    out
}

/// Sample one chance outcome via the numpy-replica RNG, EXACTLY as the Python
/// oracle does: r = rng.random(); walk cumulative probabilities; pick the first
/// outcome whose cumulative prob >= r (default to the LAST outcome).
/// (Mirrors async_mcts._expand_and_evaluate and play_one_async_game.)
pub fn sample_chance(engine: &Engine, rng: &mut NpRng) -> u32 {
    let outcomes = engine.chance_outcomes(); // Vec<(u32, f64)>
    let r = rng.random_f64();
    let mut cum = 0.0f64;
    let mut chosen = outcomes.last().expect("chance node has no outcomes").0;
    for &(v, p) in &outcomes {
        cum += p;
        if r <= cum {
            chosen = v;
            break;
        }
    }
    chosen
}

/// Public VP of seat `p` (arena tiebreak input).
pub fn vp(engine: &Engine, p: usize) -> u8 {
    engine.state.vp[p]
}

/// VP margin = top - second (arena's skill-vs-luck signal).
pub fn vp_margin(engine: &Engine) -> i32 {
    let mut vps: Vec<i32> = (0..N_PLAYERS).map(|p| engine.state.vp[p] as i32).collect();
    vps.sort_unstable_by(|a, b| b.cmp(a));
    vps[0] - vps[1]
}

/// VP-leader tiebreak winner SEAT, matching arena._vp_leader_margin:
/// top public VP; ties broken by settlements+cities built; then -1 (draw).
pub fn vp_leader_margin(engine: &Engine) -> i32 {
    let vps: Vec<i32> = (0..N_PLAYERS).map(|p| engine.state.vp[p] as i32).collect();
    let top = *vps.iter().max().unwrap();
    let leaders: Vec<usize> = (0..N_PLAYERS).filter(|&i| vps[i] == top).collect();
    if leaders.len() == 1 {
        return leaders[0] as i32;
    }
    let stats = engine.stats();
    let builds = |i: usize| -> u32 {
        stats.players[i].settlements_built as u32 + stats.players[i].cities_built as u32
    };
    let best = leaders.iter().map(|&i| builds(i)).max().unwrap();
    let tied: Vec<usize> = leaders.into_iter().filter(|&i| builds(i) == best).collect();
    if tied.len() == 1 {
        tied[0] as i32
    } else {
        -1
    }
}

/// Replay a recorded action/chance sequence on a fresh engine (test helper +
/// the basis for serialize/deserialize parity). `entries` is (is_chance, id).
pub fn replay(seed: u64, vp_target: u8, bonuses: bool, entries: &[(bool, u32)]) -> Engine {
    let mut engine = if vp_target == 10 && bonuses {
        Engine::new(seed)
    } else {
        Engine::with_rules(seed, vp_target, bonuses)
    };
    for &(is_chance, id) in entries {
        if is_chance {
            engine.apply_chance_outcome(id);
        } else {
            engine.step(id);
        }
    }
    engine
}
