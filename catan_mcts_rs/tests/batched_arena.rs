//! Task 2: two-queue cross-game batched arena scheduler.
//!
//! (a) reproducibility: running the same (pairs, both-batched-evaluators) twice
//! yields field-identical ArenaGameResult vectors (mirrors batched_selfplay.rs's
//! records_equal contract).
//! (b) agreement vs the B=1 oracle (`play_arena_game`): batched kernels
//! reassociate floats ~1e-7, so a rare argmax flip is tolerated at padding
//! boundaries, but a systematic divergence is not — require >=7/8 winner_seat
//! matches.

use catan_mcts_rs::arena::{play_arena_game, seating_is_cand, seed_plan, ArenaGameResult};
use catan_mcts_rs::arena::play_arena_games_batched;
use catan_mcts_rs::evaluator::TorchScriptEvaluator;
use std::path::PathBuf;
use tch::Device;

fn spike() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("mcts_study").join("spike")
}

fn results_equal(a: &ArenaGameResult, b: &ArenaGameResult) -> bool {
    a.winner_seat == b.winner_seat && a.timed_out == b.timed_out && a.vp_margin == b.vp_margin
}

#[test]
fn batched_arena_reproducible() {
    let ts = spike().join("wrapper_batched.ts");
    if !ts.exists() {
        eprintln!("skip: {ts:?} missing (run scripts/export_spike_batched.py)");
        return;
    }
    let ev_cand = TorchScriptEvaluator::load_batched(ts.to_str().unwrap(), Device::Cpu, 8).unwrap();
    let ev_champ = TorchScriptEvaluator::load_batched(ts.to_str().unwrap(), Device::Cpu, 8).unwrap();
    let pairs = seed_plan(7000, 8);

    let run_a = play_arena_games_batched(&ev_cand, &ev_champ, &pairs, 8, 10, true, 5000);
    let run_b = play_arena_games_batched(&ev_cand, &ev_champ, &pairs, 8, 10, true, 5000);

    assert_eq!(run_a.len(), pairs.len());
    assert_eq!(run_b.len(), pairs.len());
    for (i, (a, b)) in run_a.iter().zip(run_b.iter()).enumerate() {
        assert!(results_equal(a, b), "batched arena NOT reproducible at index {i} (rot={}, seed={})", pairs[i].0, pairs[i].1);
    }
}

#[test]
fn batched_arena_agreement_vs_oracle() {
    let ts_batched = spike().join("wrapper_batched.ts");
    let ts_traced = spike().join("wrapper_traced.ts");
    if !ts_batched.exists() || !ts_traced.exists() {
        eprintln!("skip: fixtures missing ({ts_batched:?}, {ts_traced:?})");
        return;
    }
    let ev_cand =
        TorchScriptEvaluator::load_batched(ts_batched.to_str().unwrap(), Device::Cpu, 8).unwrap();
    let ev_champ =
        TorchScriptEvaluator::load_batched(ts_batched.to_str().unwrap(), Device::Cpu, 8).unwrap();
    let ev_oracle = TorchScriptEvaluator::load(ts_traced.to_str().unwrap(), Device::Cpu).unwrap();

    let pairs = seed_plan(7000, 8);
    let batched = play_arena_games_batched(&ev_cand, &ev_champ, &pairs, 8, 10, true, 5000);

    let mut matches = 0;
    let mut mismatches = Vec::new();
    for (i, &(rot, seed)) in pairs.iter().enumerate() {
        let seating = seating_is_cand(rot);
        let oracle = play_arena_game(&ev_oracle, &ev_oracle, seed, seating, 8, 10, true, 5000);
        if batched[i].winner_seat == oracle.winner_seat {
            matches += 1;
        } else {
            mismatches.push((rot, seed, batched[i].winner_seat, oracle.winner_seat));
        }
    }

    if !mismatches.is_empty() {
        eprintln!("mismatches (rot, seed, batched_winner, oracle_winner): {mismatches:?}");
    }
    assert!(matches >= 7, "only {matches}/8 winner_seat matches; mismatches: {mismatches:?}");
}
