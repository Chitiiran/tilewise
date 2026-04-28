//! Throughput bench WITHOUT the 5000-step cap, to measure actual full-game cost
//! for the chance-aware engine (post-Phase-0). This is a one-off bench to inform
//! the MCTS-study compute budgeting.

use catan_engine::Engine;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn play_full_game(seed: u64) -> u32 {
    let mut engine = Engine::new(seed);
    let mut steps = 0u32;
    while !engine.is_terminal() {
        let legal = engine.legal_actions();
        if legal.is_empty() { break; }
        engine.step(legal[(steps as usize) % legal.len()]);
        steps += 1;
        if steps > 100_000 { break; }
    }
    steps
}

fn bench_full(c: &mut Criterion) {
    c.bench_function("random_full_game_seed_42", |b| {
        b.iter(|| play_full_game(black_box(42)));
    });
}

criterion_group!(benches, bench_full);
criterion_main!(benches);
