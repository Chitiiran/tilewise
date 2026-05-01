//! v2 playstrength quick-check: drive 100 games with greedy-priority moves,
//! measure timeouts, length distribution, VP-source breakdown.
//!
//! Compare to v1_playstrength.json (41% timeouts, median 4648 moves).

use catan_engine::Engine;
use std::time::Instant;

fn pick_greedy(legal: &[u32]) -> u32 {
    fn priority(a: u32) -> i32 {
        match a {
            54..=107 => 100,   // city
            0..=53 => 80,      // settlement
            108..=179 => 50,   // road
            206..=225 => 30,   // trade-bank (use when can't build, beats EndTurn)
            205 => 10,         // roll dice
            180..=198 => 5,    // robber
            199..=203 => 5,    // discard (deadcode)
            204 => 1,          // end turn (last resort)
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

fn play_one_game(seed: u64, max_steps: u32) -> (Option<u8>, u32, [u8; 4], Option<u8>) {
    let mut engine = Engine::new(seed);
    let mut steps = 0u32;
    while !engine.is_terminal() && steps < max_steps {
        if engine.is_chance_pending() {
            let outcomes = engine.chance_outcomes();
            if outcomes.is_empty() { break; }
            engine.apply_chance_outcome(outcomes[0].0);
        } else {
            let legal = engine.legal_actions();
            if legal.is_empty() { break; }
            engine.step(pick_greedy(&legal));
        }
        steps += 1;
    }
    let winner = if engine.is_terminal() {
        let stats = engine.stats();
        let w = stats.winner_player_id;
        if w >= 0 { Some(w as u8) } else { None }
    } else { None };
    (winner, steps, engine.state.vp, engine.state.longest_road_holder)
}

fn main() {
    const N_GAMES: usize = 100;
    const MAX_STEPS: u32 = 30_000;

    let mut lengths = Vec::with_capacity(N_GAMES);
    let mut timeouts = 0u32;
    let mut wins_by_seat = [0u32; 4];
    let mut had_longest_road = 0u32;
    let mut total_winner_vp = 0u32;

    let start = Instant::now();
    let mut sample_vp = [0u8; 4];
    let mut sample_lr_lengths = [0u8; 4];
    for seed in 0..N_GAMES as u64 {
        let (winner, steps, vp, lr_holder) = play_one_game(seed, MAX_STEPS);
        lengths.push(steps);
        if let Some(w) = winner {
            wins_by_seat[w as usize] += 1;
            total_winner_vp += vp[w as usize] as u32;
        } else {
            timeouts += 1;
            if seed == 0 {
                sample_vp = vp;
                // Re-run to get lr lengths since state isn't preserved.
                let mut e = Engine::new(seed);
                let mut s = 0u32;
                while !e.is_terminal() && s < MAX_STEPS {
                    if e.is_chance_pending() {
                        let outcomes = e.chance_outcomes();
                        if outcomes.is_empty() { break; }
                        e.apply_chance_outcome(outcomes[0].0);
                    } else {
                        let legal = e.legal_actions();
                        if legal.is_empty() { break; }
                        e.step(pick_greedy(&legal));
                    }
                    s += 1;
                }
                sample_lr_lengths = e.state.longest_road_length;
            }
        }
        if lr_holder.is_some() { had_longest_road += 1; }
    }
    eprintln!("Sample seed=0 (timeout): vp={:?} lr_lengths={:?}", sample_vp, sample_lr_lengths);
    let elapsed = start.elapsed().as_secs_f64();

    lengths.sort();
    let p25 = lengths[N_GAMES / 4];
    let median = lengths[N_GAMES / 2];
    let p75 = lengths[3 * N_GAMES / 4];

    let n_finished = (N_GAMES - timeouts as usize) as u32;
    let mean_winner_vp = total_winner_vp as f64 / n_finished.max(1) as f64;

    println!(r#"{{"version":"v2_phase1","n_games":{},"timeouts":{},"timeout_rate":{:.4},"length_p25":{},"length_median":{},"length_p75":{},"length_max":{},"wins_by_seat":[{},{},{},{}],"games_with_longest_road":{},"mean_winner_vp":{:.2},"elapsed_sec":{:.1}}}"#,
        N_GAMES, timeouts, timeouts as f64 / N_GAMES as f64,
        p25, median, p75, lengths[N_GAMES - 1],
        wins_by_seat[0], wins_by_seat[1], wins_by_seat[2], wins_by_seat[3],
        had_longest_road, mean_winner_vp, elapsed,
    );
}
