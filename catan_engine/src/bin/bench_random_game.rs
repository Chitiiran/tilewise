//! Profile a full all-random game to find where time is going.
//!
//! All players make uniformly-random legal moves. Times each engine call
//! category separately so we can see whether the bottleneck is
//! legal_actions, step, chance_outcomes, apply_chance_outcome, or other.
//!
//! Output (one JSON line per category, plus a summary):
//!     {"workload": "rand-legal_actions", "calls": N, "total_us": ..., "mean_ns": ...}
//!     ...
//!     {"workload": "rand-game-summary", "n_games": G, "total_us": T,
//!      "mean_us_per_game": ..., "mean_steps_per_game": ..., "wins": [...], "no_winner": K}
//!
//! Usage:
//!     cargo run --release --bin bench_random_game -- --n-games 100 --seed-base 1000
use catan_engine::Engine;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::env;
use std::time::{Duration, Instant};

#[derive(Default)]
struct Profile {
    legal_actions_calls: u64,
    legal_actions_time: Duration,
    step_calls: u64,
    step_time: Duration,
    chance_outcomes_calls: u64,
    chance_outcomes_time: Duration,
    apply_chance_calls: u64,
    apply_chance_time: Duration,
    is_terminal_calls: u64,
    is_terminal_time: Duration,
    is_chance_pending_calls: u64,
    is_chance_pending_time: Duration,
    legal_mask_calls: u64,
    legal_mask_time: Duration,
    rng_pick_time: Duration,
    total_steps: u64,
    games_completed: u64,
    games_no_winner: u64,
    wins: [u64; 4],
    total_game_time: Duration,
}

impl Profile {
    fn merge(&mut self, other: &Profile) {
        self.legal_actions_calls += other.legal_actions_calls;
        self.legal_actions_time += other.legal_actions_time;
        self.step_calls += other.step_calls;
        self.step_time += other.step_time;
        self.chance_outcomes_calls += other.chance_outcomes_calls;
        self.chance_outcomes_time += other.chance_outcomes_time;
        self.apply_chance_calls += other.apply_chance_calls;
        self.apply_chance_time += other.apply_chance_time;
        self.is_terminal_calls += other.is_terminal_calls;
        self.is_terminal_time += other.is_terminal_time;
        self.is_chance_pending_calls += other.is_chance_pending_calls;
        self.is_chance_pending_time += other.is_chance_pending_time;
        self.legal_mask_calls += other.legal_mask_calls;
        self.legal_mask_time += other.legal_mask_time;
        self.rng_pick_time += other.rng_pick_time;
        self.total_steps += other.total_steps;
        self.games_completed += other.games_completed;
        self.games_no_winner += other.games_no_winner;
        for i in 0..4 {
            self.wins[i] += other.wins[i];
        }
        self.total_game_time += other.total_game_time;
    }
}

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

fn play_one_game(seed: u64, max_steps: u64, prof: &mut Profile) {
    let mut e = Engine::new(seed);
    let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(0xDEAD_BEEF));
    let mut steps: u64 = 0;
    let game_start = Instant::now();

    loop {
        // is_terminal()
        let t0 = Instant::now();
        let term = e.is_terminal();
        prof.is_terminal_time += t0.elapsed();
        prof.is_terminal_calls += 1;
        if term || steps >= max_steps {
            break;
        }
        // is_chance_pending()
        let t0 = Instant::now();
        let chance = e.is_chance_pending();
        prof.is_chance_pending_time += t0.elapsed();
        prof.is_chance_pending_calls += 1;

        if chance {
            // chance_outcomes()
            let t0 = Instant::now();
            let outcomes = e.chance_outcomes();
            prof.chance_outcomes_time += t0.elapsed();
            prof.chance_outcomes_calls += 1;
            if outcomes.is_empty() {
                break;
            }
            // pick one by cum prob (counted under rng_pick_time)
            let t0 = Instant::now();
            let r: f64 = rng.gen();
            let chosen = sample_chance(&outcomes, r);
            prof.rng_pick_time += t0.elapsed();
            // apply_chance_outcome()
            let t0 = Instant::now();
            e.apply_chance_outcome(chosen);
            prof.apply_chance_time += t0.elapsed();
            prof.apply_chance_calls += 1;
        } else {
            // legal_actions()
            let t0 = Instant::now();
            let legal = e.legal_actions();
            prof.legal_actions_time += t0.elapsed();
            prof.legal_actions_calls += 1;
            if legal.is_empty() {
                break;
            }
            // pick one uniformly
            let t0 = Instant::now();
            let i = rng.gen_range(0..legal.len());
            let action = legal[i];
            prof.rng_pick_time += t0.elapsed();
            // step()
            let t0 = Instant::now();
            e.step(action);
            prof.step_time += t0.elapsed();
            prof.step_calls += 1;
        }
        steps += 1;
    }
    prof.total_steps += steps;
    prof.total_game_time += game_start.elapsed();

    if e.is_terminal() {
        let stats = e.stats();
        let w = stats.winner_player_id;
        if (0..4).contains(&w) {
            prof.wins[w as usize] += 1;
            prof.games_completed += 1;
        } else {
            prof.games_no_winner += 1;
        }
    } else {
        prof.games_no_winner += 1;
    }
}

fn print_category(name: &str, calls: u64, time: Duration) {
    let total_us = time.as_secs_f64() * 1_000_000.0;
    let mean_ns = if calls > 0 {
        time.as_nanos() as f64 / calls as f64
    } else {
        0.0
    };
    println!(
        r#"{{"workload":"{}","calls":{},"total_us":{:.3},"mean_ns":{:.3}}}"#,
        name, calls, total_us, mean_ns
    );
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut n_games: u64 = 100;
    let mut seed_base: u64 = 1000;
    let mut max_steps: u64 = 30_000;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--n-games" => {
                n_games = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--seed-base" => {
                seed_base = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--max-steps" => {
                max_steps = args[i + 1].parse().unwrap();
                i += 2;
            }
            _ => i += 1,
        }
    }
    eprintln!(
        "[bench-random-game] n_games={} seed_base={} max_steps={}",
        n_games, seed_base, max_steps
    );

    let mut prof = Profile::default();
    let total_start = Instant::now();
    for g in 0..n_games {
        let mut p = Profile::default();
        play_one_game(seed_base + g, max_steps, &mut p);
        prof.merge(&p);
    }
    let total = total_start.elapsed();

    print_category("rand-legal_actions", prof.legal_actions_calls, prof.legal_actions_time);
    print_category("rand-step", prof.step_calls, prof.step_time);
    print_category("rand-chance_outcomes", prof.chance_outcomes_calls, prof.chance_outcomes_time);
    print_category("rand-apply_chance", prof.apply_chance_calls, prof.apply_chance_time);
    print_category("rand-is_terminal", prof.is_terminal_calls, prof.is_terminal_time);
    print_category("rand-is_chance_pending", prof.is_chance_pending_calls, prof.is_chance_pending_time);
    print_category("rand-rng_pick", prof.total_steps, prof.rng_pick_time);

    let total_us = total.as_secs_f64() * 1_000_000.0;
    let game_total_us = prof.total_game_time.as_secs_f64() * 1_000_000.0;
    let mean_us_per_game = game_total_us / n_games as f64;
    let mean_steps_per_game = prof.total_steps as f64 / n_games as f64;
    let mean_us_per_step = if prof.total_steps > 0 {
        game_total_us / prof.total_steps as f64
    } else {
        0.0
    };
    println!(
        r#"{{"workload":"rand-game-summary","n_games":{},"total_us":{:.3},"game_total_us":{:.3},"mean_us_per_game":{:.3},"mean_steps_per_game":{:.3},"mean_us_per_step":{:.3},"wins":[{},{},{},{}],"games_completed":{},"games_no_winner":{}}}"#,
        n_games, total_us, game_total_us, mean_us_per_game, mean_steps_per_game,
        mean_us_per_step,
        prof.wins[0], prof.wins[1], prof.wins[2], prof.wins[3],
        prof.games_completed, prof.games_no_winner,
    );
}
