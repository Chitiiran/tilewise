//! Task 10 THROUGHPUT COMPARISON (ignored by default; run explicitly).
//! Times B=1 per-game self-play vs batched multi-game self-play on the spike
//! (production 128x4) net, same seeds, same sims. Prints games/min for each.
//!
//! Run: cargo test -p catan_mcts_rs --test throughput_compare -- --ignored --nocapture

use catan_mcts_rs::evaluator::TorchScriptEvaluator;
use catan_mcts_rs::selfplay::{play_games_batched, play_one_game, SelfPlayConfig};
use std::path::PathBuf;
use std::time::Instant;
use tch::Device;

fn spike() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("mcts_study").join("spike")
}

#[test]
#[ignore]
fn compare_b1_vs_batched_throughput() {
    let device = if tch::Cuda::is_available() { Device::Cuda(0) } else { Device::Cpu };
    eprintln!("device = {:?}", device);

    // b_max + game count come from env so we can measure the production config
    // (B_MAX=32, 32 concurrent games) without recompiling.
    let single = spike().join("wrapper_traced.ts");
    let batched = spike().join("wrapper_batched.ts");
    let b_max: usize = std::env::var("TP_BMAX").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    let n_games: u64 = std::env::var("TP_GAMES").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    let n_sims: u32 = std::env::var("TP_SIMS").ok().and_then(|v| v.parse().ok()).unwrap_or(200);
    let seeds: Vec<u64> = (0..n_games).collect();
    let cfg = SelfPlayConfig { n_sims, self_play: true, max_steps: 200_000, ..Default::default() };

    let skip_b1 = std::env::var("TP_SKIP_B1").is_ok();
    eprintln!("config: b_max={b_max} games={n_games} sims={n_sims} skip_b1={skip_b1}");
    let n = seeds.len() as f64;

    // B=1 per-game (skippable for big runs — slow baseline already measured).
    if !skip_b1 {
        let ev1 = TorchScriptEvaluator::load(single.to_str().unwrap(), device).unwrap();
        let t0 = Instant::now();
        let mut moves_b1 = 0usize;
        for &s in &seeds {
            moves_b1 += play_one_game(&ev1, s, &cfg).moves.len();
        }
        let dt_b1 = t0.elapsed().as_secs_f64();
        eprintln!("B=1     : {dt_b1:.1}s for {} games -> {:.2} games/min ({moves_b1} moves)",
                  seeds.len(), n / dt_b1 * 60.0);
    }

    // Batched.
    let evb = TorchScriptEvaluator::load_batched(batched.to_str().unwrap(), device, b_max).unwrap();
    let t1 = Instant::now();
    let rb = play_games_batched(&evb, &seeds, &cfg);
    let dt_bt = t1.elapsed().as_secs_f64();
    let moves_bt: usize = rb.iter().map(|r| r.moves.len()).sum();
    eprintln!("batched : {dt_bt:.1}s for {} games -> {:.2} games/min ({moves_bt} moves)",
              seeds.len(), n / dt_bt * 60.0);
}
