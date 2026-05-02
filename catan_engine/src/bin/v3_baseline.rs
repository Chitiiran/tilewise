//! v3.1 measurement: how much shorter are v3 games vs v2 under random play?
//!
//! Drives N games under each rule set with uniformly-random legal moves and
//! reports length distribution + wall-clock time. Calibrates the v3.4
//! validation-sweep volume target.
//!
//! Why pure-random here, not lookahead-MCTS: lookahead lives in Python via
//! PyO3 (wired up in v3.2). Pure-random in Rust runs in the same process so
//! we get a clean apples-to-apples Rust-side comparison. Lookahead numbers
//! come out of v3.4's data-gen sweep naturally.
//!
//! Run:
//!   cargo run --release --bin v3_baseline
//! Stdout is two JSON lines (one per rule set); stderr has progress.

use catan_engine::Engine;
use rand::rngs::SmallRng;
use rand::{Rng as RandRng, SeedableRng};
use std::time::Instant;

const N_GAMES: usize = 200;
const MAX_STEPS: u32 = 30_000;

/// Sample one chance outcome by cumulative probability.
#[inline]
fn sample_chance(outcomes: &[(u32, f64)], r: f64) -> u32 {
    let mut cum = 0.0;
    for &(v, p) in outcomes {
        cum += p;
        if r <= cum {
            return v;
        }
    }
    outcomes[outcomes.len() - 1].0
}

/// Returns (winner, steps_taken, final_vp, max_vp_at_end, hit_step_cap).
fn play_one_random_game(engine: &mut Engine, picker: &mut SmallRng) -> (Option<u8>, u32, [u8; 4], u8, bool) {
    let mut steps = 0u32;
    while !engine.is_terminal() && steps < MAX_STEPS {
        if engine.is_chance_pending() {
            let outcomes = engine.chance_outcomes();
            if outcomes.is_empty() { break; }
            let r: f64 = picker.gen();
            engine.apply_chance_outcome(sample_chance(&outcomes, r));
        } else {
            let legal = engine.legal_actions();
            if legal.is_empty() { break; }
            let idx = picker.gen_range(0..legal.len());
            engine.step(legal[idx]);
        }
        steps += 1;
    }
    let hit_cap = !engine.is_terminal() && steps >= MAX_STEPS;
    let winner = if engine.is_terminal() {
        let stats = engine.stats();
        let w = stats.winner_player_id;
        if (0..4).contains(&w) { Some(w as u8) } else { None }
    } else { None };
    let max_vp = *engine.state.vp.iter().max().unwrap_or(&0);
    (winner, steps, engine.state.vp, max_vp, hit_cap)
}

struct Summary {
    label: &'static str,
    n_games: usize,
    timeouts: u32,
    cap_hits: u32,
    lengths: Vec<u32>,
    wall_clock_sec: f64,
    wins_by_seat: [u32; 4],
    sum_winner_vp: u32,
    sum_max_vp_at_end: u32,
}

fn run_sweep(label: &'static str, vp_target: u8, bonuses: bool) -> Summary {
    let mut lengths = Vec::with_capacity(N_GAMES);
    let mut timeouts = 0u32;
    let mut cap_hits = 0u32;
    let mut wins_by_seat = [0u32; 4];
    let mut sum_winner_vp = 0u32;
    let mut sum_max_vp_at_end = 0u32;
    let start = Instant::now();
    for seed in 0..N_GAMES as u64 {
        let mut engine = Engine::with_rules(seed, vp_target, bonuses);
        // Distinct seed for the move-picker so it can't perfectly track the engine RNG.
        let mut picker = SmallRng::seed_from_u64(seed.wrapping_add(0xDEADBEEF));
        let (winner, steps, vp, max_vp, hit_cap) = play_one_random_game(&mut engine, &mut picker);
        lengths.push(steps);
        if hit_cap { cap_hits += 1; }
        match winner {
            Some(w) => {
                wins_by_seat[w as usize] += 1;
                sum_winner_vp += vp[w as usize] as u32;
            }
            None => { timeouts += 1; }
        }
        sum_max_vp_at_end += max_vp as u32;
        if (seed + 1) % 50 == 0 {
            eprintln!("  [{}] {}/{} games done", label, seed + 1, N_GAMES);
        }
    }
    Summary {
        label,
        n_games: N_GAMES,
        timeouts,
        cap_hits,
        lengths,
        wall_clock_sec: start.elapsed().as_secs_f64(),
        wins_by_seat,
        sum_winner_vp,
        sum_max_vp_at_end,
    }
}

fn print_summary(s: &Summary) {
    let mut sorted = s.lengths.clone();
    sorted.sort();
    let n = sorted.len();
    let p25 = sorted[n / 4];
    let median = sorted[n / 2];
    let p75 = sorted[3 * n / 4];
    let max = sorted[n - 1];
    let mean: f64 = sorted.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
    let n_finished = (n - s.timeouts as usize).max(1) as u32;
    let mean_winner_vp = s.sum_winner_vp as f64 / n_finished as f64;
    let mean_max_vp = s.sum_max_vp_at_end as f64 / n as f64;
    let games_per_sec = n as f64 / s.wall_clock_sec;
    let ms_per_game = s.wall_clock_sec * 1000.0 / n as f64;

    println!(
        r#"{{"label":"{}","n_games":{},"timeouts":{},"cap_hits":{},"length_p25":{},"length_median":{},"length_mean":{:.1},"length_p75":{},"length_max":{},"wins_by_seat":[{},{},{},{}],"mean_winner_vp":{:.2},"mean_max_vp_at_end":{:.2},"wall_clock_sec":{:.2},"ms_per_game":{:.1},"games_per_sec":{:.1}}}"#,
        s.label, s.n_games, s.timeouts, s.cap_hits,
        p25, median, mean, p75, max,
        s.wins_by_seat[0], s.wins_by_seat[1], s.wins_by_seat[2], s.wins_by_seat[3],
        mean_winner_vp, mean_max_vp,
        s.wall_clock_sec, ms_per_game, games_per_sec,
    );
}

fn main() {
    eprintln!("v3_baseline: running {} random games per rule set...", N_GAMES);

    eprintln!("[1/2] v2 rules (vp_target=10, bonuses=true)");
    let v2 = run_sweep("v2", 10, true);

    eprintln!("[2/2] v3 rules (vp_target=5, bonuses=false)");
    let v3 = run_sweep("v3", 5, false);

    print_summary(&v2);
    print_summary(&v3);
}
