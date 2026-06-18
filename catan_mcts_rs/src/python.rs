//! PyO3 surface for catan_mcts_rs.
//!
//! Production entry points (`run_selfplay`, `run_arena`) land here in later
//! phases — coarse, one crossing per STAGE. For now this exposes the Phase-4
//! `debug_*` hooks the differential parity test drives (replay an action
//! sequence on a fresh engine, return a queried value). These hooks stay (test
//! surface) but are not on the hot path.

use crate::arena::{play_arena_game, seating_is_cand};
use crate::evaluator::TorchScriptEvaluator;
use crate::mcts::{best_action, Mcts};
use crate::rng::NpRng;
use crate::selfplay::{play_one_game, SelfPlayConfig};
use crate::state;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use tch::Device;

/// Parse a Python list of (is_chance: bool, id: int) into replay entries.
fn parse_entries(entries: Vec<(bool, u32)>) -> Vec<(bool, u32)> {
    entries
}

/// Replay `entries` then return legal_actions() as a sorted Vec.
#[pyfunction]
#[pyo3(signature = (seed, entries, vp_target=10, bonuses=true))]
fn debug_legal_actions(
    seed: u64,
    entries: Vec<(bool, u32)>,
    vp_target: u8,
    bonuses: bool,
) -> Vec<u32> {
    let mut e = state::replay(seed, vp_target, bonuses, &parse_entries(entries));
    e.legal_actions()
}

/// Replay then return (is_terminal, is_chance_pending, current_player_mapped).
#[pyfunction]
#[pyo3(signature = (seed, entries, vp_target=10, bonuses=true))]
fn debug_status(
    seed: u64,
    entries: Vec<(bool, u32)>,
    vp_target: u8,
    bonuses: bool,
) -> (bool, bool, i32) {
    let e = state::replay(seed, vp_target, bonuses, &parse_entries(entries));
    (e.is_terminal(), e.is_chance_pending(), state::current_player(&e))
}

/// Replay then return the absolute-seat returns vector.
#[pyfunction]
#[pyo3(signature = (seed, entries, vp_target=10, bonuses=true))]
fn debug_returns(
    seed: u64,
    entries: Vec<(bool, u32)>,
    vp_target: u8,
    bonuses: bool,
) -> Vec<f32> {
    let e = state::replay(seed, vp_target, bonuses, &parse_entries(entries));
    state::returns_abs(&e).to_vec()
}

/// Replay then return chance_outcomes() as (value, prob) pairs.
#[pyfunction]
#[pyo3(signature = (seed, entries, vp_target=10, bonuses=true))]
fn debug_chance_outcomes(
    seed: u64,
    entries: Vec<(bool, u32)>,
    vp_target: u8,
    bonuses: bool,
) -> Vec<(u32, f64)> {
    let e = state::replay(seed, vp_target, bonuses, &parse_entries(entries));
    e.chance_outcomes()
}

/// Replay then return the four VPs and the action_history.
#[pyfunction]
#[pyo3(signature = (seed, entries, vp_target=10, bonuses=true))]
fn debug_vps_and_history(
    seed: u64,
    entries: Vec<(bool, u32)>,
    vp_target: u8,
    bonuses: bool,
) -> (Vec<u8>, Vec<u32>) {
    let e = state::replay(seed, vp_target, bonuses, &parse_entries(entries));
    let vps: Vec<u8> = (0..4).map(|p| state::vp(&e, p)).collect();
    (vps, e.action_history().to_vec())
}

/// Phase-5 parity hook: replay `entries`, then run MCTS with the given net,
/// rng_seed, sims, and Dirichlet params. Returns (visit_counts[280],
/// best_action, root_value). Mirrors AsyncMcts(...).search.
#[pyfunction]
#[pyo3(signature = (net_ts, seed, entries, n_sims, rng_seed, dirichlet_eps=0.0,
                    dirichlet_alpha=0.8, c=1.4, vp_target=10, bonuses=true))]
#[allow(clippy::too_many_arguments)]
fn debug_search(
    net_ts: String,
    seed: u64,
    entries: Vec<(bool, u32)>,
    n_sims: u32,
    rng_seed: u64,
    dirichlet_eps: f64,
    dirichlet_alpha: f64,
    c: f64,
    vp_target: u8,
    bonuses: bool,
) -> (Vec<i32>, usize, f64) {
    let engine = state::replay(seed, vp_target, bonuses, &parse_entries(entries));
    let ev = TorchScriptEvaluator::load(&net_ts, Device::Cpu).expect("load net_ts");
    let mut rng = NpRng::from_seed(rng_seed);
    let mut mcts = Mcts::new(&ev, c, dirichlet_alpha, dirichlet_eps);
    let vc = mcts.search(engine, n_sims, &mut rng);
    let ba = best_action(&vc);
    (vc.to_vec(), ba, mcts.last_root_value)
}

/// Phase-6 hook: run ONE self-play (or greedy) game and return its record as a
/// dict matching the SelfPlayRecorder fields (winner, final_vp, length,
/// action_history, moves[]). The Python recorder writes the parquet; this keeps
/// the schema identical by construction.
#[pyfunction]
#[pyo3(signature = (net_ts, seed, n_sims, self_play, vp_target=10, bonuses=true,
                    temp_moves=30, dirichlet_alpha=0.8, dirichlet_eps=0.25,
                    max_steps=200_000))]
#[allow(clippy::too_many_arguments)]
fn debug_selfplay_game<'py>(
    py: Python<'py>,
    net_ts: String,
    seed: u64,
    n_sims: u32,
    self_play: bool,
    vp_target: u8,
    bonuses: bool,
    temp_moves: usize,
    dirichlet_alpha: f64,
    dirichlet_eps: f64,
    max_steps: u32,
) -> PyResult<Bound<'py, PyDict>> {
    let ev = TorchScriptEvaluator::load(&net_ts, Device::Cpu).expect("load net_ts");
    let cfg = SelfPlayConfig {
        n_sims,
        self_play,
        vp_target,
        bonuses,
        max_steps,
        temp_moves,
        dirichlet_alpha,
        dirichlet_eps,
    };
    let rec = play_one_game(&ev, seed, &cfg);
    game_record_to_dict(py, &rec)
}

/// Convert a Rust GameRecord into the SelfPlayRecorder-shaped Python dict.
fn game_record_to_dict<'py>(
    py: Python<'py>,
    rec: &crate::selfplay::GameRecord,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new_bound(py);
    d.set_item("seed", rec.seed)?;
    d.set_item("terminal", rec.terminal)?;
    d.set_item("winner", rec.winner)?;
    d.set_item("final_vp", rec.final_vp.to_vec())?;
    d.set_item("length_in_moves", rec.length_in_moves)?;
    d.set_item("action_history", rec.action_history.clone())?;
    let moves = pyo3::types::PyList::empty_bound(py);
    for m in &rec.moves {
        let md = PyDict::new_bound(py);
        md.set_item("current_player", m.current_player)?;
        md.set_item("move_index", m.move_index)?;
        md.set_item("action_taken", m.action_taken)?;
        md.set_item("root_value", m.root_value)?;
        md.set_item("visit_counts", m.visit_counts.clone())?;
        md.set_item(
            "legal_mask",
            m.legal_mask.iter().map(|&b| b as u8).collect::<Vec<u8>>(),
        )?;
        moves.append(md)?;
    }
    d.set_item("moves", moves)?;
    Ok(d)
}

/// Phase-7 hook: play ONE arena game (cand vs champ nets, greedy) at the given
/// rotation+seed. Returns (winner_seat, timed_out, vp_margin) — same triple as
/// arena._play_arena_game.
#[pyfunction]
#[pyo3(signature = (cand_ts, champ_ts, seed, rot, sims, vp_target=10,
                    bonuses=true, max_steps=200_000))]
#[allow(clippy::too_many_arguments)]
fn debug_arena_game(
    cand_ts: String,
    champ_ts: String,
    seed: u64,
    rot: usize,
    sims: u32,
    vp_target: u8,
    bonuses: bool,
    max_steps: u32,
) -> (i32, bool, i32) {
    let ev_cand = TorchScriptEvaluator::load(&cand_ts, Device::Cpu).expect("load cand_ts");
    let ev_champ = TorchScriptEvaluator::load(&champ_ts, Device::Cpu).expect("load champ_ts");
    let seating = seating_is_cand(rot);
    let r = play_arena_game(
        &ev_cand, &ev_champ, seed, seating, sims, vp_target, bonuses, max_steps,
    );
    (r.winner_seat, r.timed_out, r.vp_margin)
}

/// PRODUCTION self-play entry (coarse — one crossing for the whole stage).
/// Runs `seeds` games with one shared net evaluator, returns a list of records
/// (same dict shape as debug_selfplay_game) for the Python SelfPlayRecorder.
#[pyfunction]
#[pyo3(signature = (net_ts, seeds, n_sims, self_play=true, vp_target=10,
                    bonuses=true, temp_moves=30, dirichlet_alpha=0.8,
                    dirichlet_eps=0.25, max_steps=200_000))]
#[allow(clippy::too_many_arguments)]
fn run_selfplay<'py>(
    py: Python<'py>,
    net_ts: String,
    seeds: Vec<u64>,
    n_sims: u32,
    self_play: bool,
    vp_target: u8,
    bonuses: bool,
    temp_moves: usize,
    dirichlet_alpha: f64,
    dirichlet_eps: f64,
    max_steps: u32,
) -> PyResult<Bound<'py, pyo3::types::PyList>> {
    let ev = TorchScriptEvaluator::load(&net_ts, Device::Cpu).expect("load net_ts");
    let out = pyo3::types::PyList::empty_bound(py);
    for seed in seeds {
        let cfg = SelfPlayConfig {
            n_sims, self_play, vp_target, bonuses, max_steps,
            temp_moves, dirichlet_alpha, dirichlet_eps,
        };
        let rec = play_one_game(&ev, seed, &cfg);
        out.append(game_record_to_dict(py, &rec)?)?;
    }
    Ok(out)
}

/// PRODUCTION arena entry (coarse). Plays the full (rot, seed) plan with two
/// nets; returns one (seed, rot, winner_seat, winner_role, timed_out, vp_margin)
/// dict per game for the Python ArenaResult aggregation.
#[pyfunction]
#[pyo3(signature = (cand_ts, champ_ts, seed_base, games, sims, vp_target=10,
                    bonuses=true, max_steps=200_000))]
#[allow(clippy::too_many_arguments)]
fn run_arena<'py>(
    py: Python<'py>,
    cand_ts: String,
    champ_ts: String,
    seed_base: u64,
    games: usize,
    sims: u32,
    vp_target: u8,
    bonuses: bool,
    max_steps: u32,
) -> PyResult<Bound<'py, pyo3::types::PyList>> {
    let ev_cand = TorchScriptEvaluator::load(&cand_ts, Device::Cpu).expect("load cand_ts");
    let ev_champ = TorchScriptEvaluator::load(&champ_ts, Device::Cpu).expect("load champ_ts");
    let out = pyo3::types::PyList::empty_bound(py);
    for (rot, seed) in crate::arena::seed_plan(seed_base, games) {
        let seating = seating_is_cand(rot);
        let r = play_arena_game(
            &ev_cand, &ev_champ, seed, seating, sims, vp_target, bonuses, max_steps,
        );
        let role: Option<&str> = if r.winner_seat >= 0 {
            Some(if seating[r.winner_seat as usize] { "cand" } else { "champ" })
        } else {
            None
        };
        let d = PyDict::new_bound(py);
        d.set_item("seed", seed)?;
        d.set_item("rot", rot)?;
        d.set_item("winner_seat", r.winner_seat)?;
        d.set_item("winner_role", role)?;
        d.set_item("timed_out", r.timed_out)?;
        d.set_item("vp_margin", r.vp_margin)?;
        out.append(d)?;
    }
    Ok(out)
}

#[pymodule]
fn catan_mcts_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(debug_legal_actions, m)?)?;
    m.add_function(wrap_pyfunction!(debug_status, m)?)?;
    m.add_function(wrap_pyfunction!(debug_returns, m)?)?;
    m.add_function(wrap_pyfunction!(debug_chance_outcomes, m)?)?;
    m.add_function(wrap_pyfunction!(debug_vps_and_history, m)?)?;
    m.add_function(wrap_pyfunction!(debug_search, m)?)?;
    m.add_function(wrap_pyfunction!(debug_selfplay_game, m)?)?;
    m.add_function(wrap_pyfunction!(debug_arena_game, m)?)?;
    m.add_function(wrap_pyfunction!(run_selfplay, m)?)?;
    m.add_function(wrap_pyfunction!(run_arena, m)?)?;
    Ok(())
}
