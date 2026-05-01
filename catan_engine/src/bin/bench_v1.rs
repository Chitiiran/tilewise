//! v1 engine microbench harness — writes JSONL to stdout for archival comparison vs v2.
//!
//! Workloads (per the v2 design doc §55a):
//!   bench-engine-step     — apply random legal actions, time per step
//!   bench-mcts-game       — play full game with greedy moves, total wall-clock
//!   bench-evaluator-leaf  — time one lookahead_vp_value(depth=10) call
//!   bench-state-clone     — time engine.clone() (engine wraps GameState)
//!
//! `bench-cache-build` is intentionally omitted from this harness — it lives
//! Python-side and is dominated by PyO3 boundary cost, not engine work.
//!
//! Usage:
//!     cargo run --release --bin bench_v1 -- --version v1 --git-sha $(git rev-parse HEAD) > out.jsonl
//!
//! Each line is one JSON object:
//!     {"workload": "...", "version": "v1", "git_sha": "...", "n_iters": N,
//!      "mean_us": M, "p50_us": ..., "p99_us": ...}

use catan_engine::Engine;
use std::env;
use std::time::Instant;

fn percentile(samples: &mut [f64], q: f64) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((samples.len() - 1) as f64 * q).round() as usize;
    samples[idx]
}

fn print_result(workload: &str, version: &str, git_sha: &str, samples_us: Vec<f64>) {
    let mut s = samples_us;
    let n = s.len();
    let mean = s.iter().sum::<f64>() / n as f64;
    let p50 = percentile(&mut s, 0.50);
    let p99 = percentile(&mut s, 0.99);
    let min = s[0];
    let max = s[s.len() - 1];
    println!(
        r#"{{"workload":"{}","version":"{}","git_sha":"{}","n_iters":{},"mean_us":{:.3},"p50_us":{:.3},"p99_us":{:.3},"min_us":{:.3},"max_us":{:.3}}}"#,
        workload, version, git_sha, n, mean, p50, p99, min, max,
    );
}

/// bench-engine-step: apply 10000 random legal (or chance) actions, measure wall-clock per step.
fn bench_engine_step(version: &str, git_sha: &str) {
    const N_STEPS: usize = 10_000;
    const N_RUNS: usize = 5;
    let mut samples_us = Vec::with_capacity(N_RUNS);
    for run in 0..N_RUNS {
        let mut engine = Engine::new(1000 + run as u64);
        let start = Instant::now();
        let mut steps = 0;
        while steps < N_STEPS && !engine.is_terminal() {
            if engine.is_chance_pending() {
                let outcomes = engine.chance_outcomes();
                if outcomes.is_empty() { break; }
                let chosen = outcomes[steps % outcomes.len()].0;
                engine.apply_chance_outcome(chosen);
            } else {
                let legal = engine.legal_actions();
                if legal.is_empty() { break; }
                engine.step(legal[steps % legal.len()]);
            }
            steps += 1;
        }
        let elapsed = start.elapsed().as_secs_f64() * 1_000_000.0;
        let per_step = elapsed / steps.max(1) as f64;
        samples_us.push(per_step);
    }
    print_result("bench-engine-step", version, git_sha, samples_us);
}

/// bench-mcts-game: play one full game with greedy-priority moves to terminal.
/// (Stand-in for "lightweight game play" since real MCTS is OpenSpiel-side.)
fn bench_mcts_game(version: &str, git_sha: &str) {
    const N_RUNS: usize = 10;
    let mut samples_us = Vec::with_capacity(N_RUNS);
    for run in 0..N_RUNS {
        let mut engine = Engine::new(2000 + run as u64);
        let start = Instant::now();
        let mut steps = 0;
        let max_steps = 30_000;
        while !engine.is_terminal() && steps < max_steps {
            if engine.is_chance_pending() {
                let outcomes = engine.chance_outcomes();
                if outcomes.is_empty() { break; }
                let chosen = outcomes[0].0;
                engine.apply_chance_outcome(chosen);
            } else {
                let legal = engine.legal_actions();
                if legal.is_empty() { break; }
                // Greedy: pick highest-priority action (city > settlement > road > end_turn).
                let action = pick_greedy(&legal);
                engine.step(action);
            }
            steps += 1;
        }
        let elapsed = start.elapsed().as_secs_f64() * 1_000_000.0;
        samples_us.push(elapsed);
    }
    print_result("bench-mcts-game", version, git_sha, samples_us);
}

fn pick_greedy(legal: &[u32]) -> u32 {
    fn priority(a: u32) -> i32 {
        match a {
            54..=107 => 100,
            0..=53 => 80,
            108..=179 => 50,
            205 => 10,
            204 => 1,
            180..=198 => 5,
            199..=203 => 5,
            _ => 0,
        }
    }
    let mut best = legal[0];
    let mut best_pri = priority(best);
    for &a in &legal[1..] {
        let p = priority(a);
        if p > best_pri || (p == best_pri && a < best) {
            best = a;
            best_pri = p;
        }
    }
    best
}

/// bench-evaluator-leaf: time one lookahead_vp_value(depth=10) call.
/// Drives the engine to a non-terminal main-phase state first.
fn bench_evaluator_leaf(version: &str, git_sha: &str) {
    const N_CALLS: usize = 200;
    // Drive forward to a steady-state mid-game position once.
    let mut base_engine = Engine::new(99);
    let mut steps = 0;
    while steps < 500 && !base_engine.is_terminal() {
        if base_engine.is_chance_pending() {
            let outcomes = base_engine.chance_outcomes();
            if outcomes.is_empty() { break; }
            base_engine.apply_chance_outcome(outcomes[0].0);
        } else {
            let legal = base_engine.legal_actions();
            if legal.is_empty() { break; }
            base_engine.step(pick_greedy(&legal));
        }
        steps += 1;
    }
    // Time the lookahead from this position.
    let mut samples_us = Vec::with_capacity(N_CALLS);
    for i in 0..N_CALLS {
        let mut e = base_engine.clone();  // clone is part of the workload — leaf eval needs to clone first
        let start = Instant::now();
        let _ = e.lookahead_vp_value(10, 12345 + i as u64);
        samples_us.push(start.elapsed().as_secs_f64() * 1_000_000.0);
    }
    print_result("bench-evaluator-leaf", version, git_sha, samples_us);
}

/// bench-state-clone: time engine.clone() in isolation. THE bottleneck per §J37.
fn bench_state_clone(version: &str, git_sha: &str) {
    const N_CLONES: usize = 100_000;
    let mut base_engine = Engine::new(7);
    // Drive forward 100 steps for a representative state size.
    for _ in 0..100 {
        if base_engine.is_terminal() { break; }
        if base_engine.is_chance_pending() {
            let outcomes = base_engine.chance_outcomes();
            if outcomes.is_empty() { break; }
            base_engine.apply_chance_outcome(outcomes[0].0);
        } else {
            let legal = base_engine.legal_actions();
            if legal.is_empty() { break; }
            base_engine.step(pick_greedy(&legal));
        }
    }
    let start = Instant::now();
    for _ in 0..N_CLONES {
        let _e = base_engine.clone();
    }
    let elapsed_us = start.elapsed().as_secs_f64() * 1_000_000.0;
    let per_clone = elapsed_us / N_CLONES as f64;
    // Fake "samples" — single mean for now. Later v2 bench can sample more granularly.
    print_result("bench-state-clone", version, git_sha, vec![per_clone; 5]);
}

/// bench-legal-mask: time engine.legal_mask() at a representative mid-game state.
/// Phase 1.2: API parity baseline. Should converge to engine.legal_actions() cost
/// today (it's just legal_actions + into-bitmap), and beat it once incremental
/// updates land in Phase 2+.
fn bench_legal_mask(version: &str, git_sha: &str) {
    const N_CALLS: usize = 100_000;
    let mut e = Engine::new(7);
    for _ in 0..100 {
        if e.is_terminal() { break; }
        if e.is_chance_pending() {
            let outcomes = e.chance_outcomes();
            if outcomes.is_empty() { break; }
            e.apply_chance_outcome(outcomes[0].0);
        } else {
            let legal = e.legal_actions();
            if legal.is_empty() { break; }
            e.step(pick_greedy(&legal));
        }
    }
    let start = Instant::now();
    let mut total_count = 0u32;
    for _ in 0..N_CALLS {
        let m = e.legal_mask();
        total_count = total_count.wrapping_add(m.count());
    }
    let elapsed_us = start.elapsed().as_secs_f64() * 1_000_000.0;
    let per_call = elapsed_us / N_CALLS as f64;
    eprintln!("[bench-legal-mask] sanity total_count={}", total_count); // prevent dead-code elim
    print_result("bench-legal-mask", version, git_sha, vec![per_call; 5]);
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut version = "v1".to_string();
    let mut git_sha = "unknown".to_string();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--version" => { version = args[i+1].clone(); i += 2; }
            "--git-sha" => { git_sha = args[i+1].clone(); i += 2; }
            _ => { i += 1; }
        }
    }
    eprintln!("[bench] running 5 workloads, version={}, git_sha={}", version, git_sha);
    bench_engine_step(&version, &git_sha);
    bench_mcts_game(&version, &git_sha);
    bench_evaluator_leaf(&version, &git_sha);
    bench_state_clone(&version, &git_sha);
    bench_legal_mask(&version, &git_sha);
    eprintln!("[bench] done");
}
