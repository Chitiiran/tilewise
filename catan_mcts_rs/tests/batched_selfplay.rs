//! Task 10: the batched multi-game scheduler is REPRODUCIBLE — running the same
//! (seeds, b_max, batched-net) twice yields byte-identical game records. This is
//! the replay contract ("go back to the match"): a given seed set deterministically
//! replays the same games. (It is NOT equal to the per-game B=1 path — the
//! batched forward reassociates ~1e-7 and graph-i output depends on batch
//! composition — which is why the batched engine is its own canonical reference.)

use catan_mcts_rs::evaluator::TorchScriptEvaluator;
use catan_mcts_rs::selfplay::{play_games_batched, GameRecord, SelfPlayConfig};
use std::path::PathBuf;
use tch::Device;

fn spike() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("mcts_study").join("spike")
}

fn records_equal(a: &GameRecord, b: &GameRecord) -> bool {
    a.seed == b.seed
        && a.terminal == b.terminal
        && a.winner == b.winner
        && a.final_vp == b.final_vp
        && a.length_in_moves == b.length_in_moves
        && a.action_history == b.action_history
        && a.moves.len() == b.moves.len()
        && a.moves.iter().zip(b.moves.iter()).all(|(m, n)| {
            m.current_player == n.current_player
                && m.move_index == n.move_index
                && m.action_taken == n.action_taken
                && m.visit_counts == n.visit_counts
                && m.legal_mask == n.legal_mask
                && m.root_value.to_bits() == n.root_value.to_bits()
        })
}

#[test]
fn batched_selfplay_is_reproducible() {
    let ts = spike().join("wrapper_batched.ts");
    if !ts.exists() {
        eprintln!("skip: {ts:?} missing (run scripts/export_spike_batched.py)");
        return;
    }
    let ev = TorchScriptEvaluator::load_batched(ts.to_str().unwrap(), Device::Cpu, 8).unwrap();
    let cfg = SelfPlayConfig { n_sims: 8, self_play: true, max_steps: 4000, ..Default::default() };
    let seeds: Vec<u64> = (0..6).collect();

    let run_a = play_games_batched(&ev, &seeds, &cfg);
    let run_b = play_games_batched(&ev, &seeds, &cfg);

    assert_eq!(run_a.len(), seeds.len());
    for (a, b) in run_a.iter().zip(run_b.iter()) {
        assert!(records_equal(a, b),
                "batched self-play NOT reproducible for seed {}", a.seed);
    }
    // Sanity: at least some games produced moves (not all instantly terminal).
    assert!(run_a.iter().any(|r| !r.moves.is_empty()));
}
