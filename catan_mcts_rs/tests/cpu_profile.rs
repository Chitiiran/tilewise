//! CPU-side cost attribution for batched self-play (Task 10 follow-up).
//! Re-implements the play_games_batched loop with phase timers to see where the
//! wall-clock goes: GPU forward (evaluate_batch) vs CPU tree/expand (provide) vs
//! game advance (advance_to_search + finish_move). Identifies the bottleneck to
//! target. Ignored by default.
//!
//! Run: cargo test -p catan_mcts_rs --release --test cpu_profile -- --ignored --nocapture

use catan_mcts_rs::evaluator::TorchScriptEvaluator;
use catan_mcts_rs::selfplay::{profile_batched, profile_batched_timed, SelfPlayConfig};
use std::path::PathBuf;
use tch::Device;

fn spike() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("mcts_study").join("spike")
}

#[test]
#[ignore]
fn profile_cpu_phases() {
    let device = if tch::Cuda::is_available() { Device::Cuda(0) } else { Device::Cpu };
    eprintln!("device = {:?}", device);
    let b_max: usize = std::env::var("TP_BMAX").ok().and_then(|v| v.parse().ok()).unwrap_or(32);
    let n_games: u64 = std::env::var("TP_GAMES").ok().and_then(|v| v.parse().ok()).unwrap_or(32);
    let n_sims: u32 = std::env::var("TP_SIMS").ok().and_then(|v| v.parse().ok()).unwrap_or(200);

    let ev = TorchScriptEvaluator::load_batched(
        spike().join("wrapper_batched.ts").to_str().unwrap(), device, b_max).unwrap();
    let cfg = SelfPlayConfig { n_sims, self_play: true, max_steps: 200_000, ..Default::default() };
    let seeds: Vec<u64> = (0..n_games).collect();

    let timed = std::env::var("TP_TIMED").is_ok();
    let prof = if timed {
        profile_batched_timed(&ev, &seeds, &cfg)
    } else {
        profile_batched(&ev, &seeds, &cfg)
    };
    eprintln!("games={n_games} b_max={b_max} sims={n_sims} timed={timed}");
    if timed {
        eprintln!("  GPU phase split (of {:.1}s GPU):", prof.gpu_s);
        eprintln!("    marshal (parallelizable) : {:.1}s ({:.1}% of total)",
                  prof.marshal_s, 100.0 * prof.marshal_s / prof.total_s);
        eprintln!("    forward_is (IRREDUCIBLE) : {:.1}s ({:.1}% of total)",
                  prof.forward_s, 100.0 * prof.forward_s / prof.total_s);
        eprintln!("    extract (parallelizable) : {:.1}s ({:.1}% of total)",
                  prof.extract_s, 100.0 * prof.extract_s / prof.total_s);
    }
    eprintln!("total           : {:.1}s", prof.total_s);
    eprintln!("  GPU forward    : {:.1}s ({:.1}%)  [{} batches, mean B={:.1}]",
              prof.gpu_s, 100.0 * prof.gpu_s / prof.total_s, prof.n_batches,
              prof.total_leaves as f64 / prof.n_batches.max(1) as f64);
    eprintln!("  CPU provide    : {:.1}s ({:.1}%)  [expand+tree+backup]",
              prof.provide_s, 100.0 * prof.provide_s / prof.total_s);
    eprintln!("  CPU advance    : {:.1}s ({:.1}%)  [chance/single-legal + finish_move]",
              prof.advance_s, 100.0 * prof.advance_s / prof.total_s);
    eprintln!("  leaves total   : {}", prof.total_leaves);
}
