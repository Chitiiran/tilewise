//! PyO3 surface for catan_mcts_rs.
//!
//! Production entry points (`run_selfplay`, `run_arena`) land here in later
//! phases — coarse, one crossing per STAGE. For now this exposes the Phase-4
//! `debug_*` hooks the differential parity test drives (replay an action
//! sequence on a fresh engine, return a queried value). These hooks stay (test
//! surface) but are not on the hot path.

use crate::state;
use pyo3::prelude::*;

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

#[pymodule]
fn catan_mcts_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(debug_legal_actions, m)?)?;
    m.add_function(wrap_pyfunction!(debug_status, m)?)?;
    m.add_function(wrap_pyfunction!(debug_returns, m)?)?;
    m.add_function(wrap_pyfunction!(debug_chance_outcomes, m)?)?;
    m.add_function(wrap_pyfunction!(debug_vps_and_history, m)?)?;
    Ok(())
}
