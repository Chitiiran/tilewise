use catan_engine::Engine;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn play_one_game(seed: u64) -> u32 {
    let mut engine = Engine::new(seed);
    let mut steps = 0u32;
    while !engine.is_terminal() {
        let legal = engine.legal_actions();
        if legal.is_empty() { break; }
        engine.step(legal[(steps as usize) % legal.len()]);
        steps += 1;
        if steps > 5000 { break; }
    }
    steps
}

fn bench_games(c: &mut Criterion) {
    c.bench_function("random_game_seed_42", |b| {
        b.iter(|| play_one_game(black_box(42)));
    });
    c.bench_function("random_game_varied_seeds", |b| {
        let mut s = 0u64;
        b.iter(|| {
            s = s.wrapping_add(1);
            play_one_game(black_box(s))
        });
    });
}

criterion_group!(benches, bench_games);
criterion_main!(benches);
