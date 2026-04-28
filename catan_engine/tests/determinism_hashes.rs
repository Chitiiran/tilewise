use catan_engine::Engine;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

fn game_hash(seed: u64) -> u64 {
    let mut engine = Engine::new(seed);
    let mut h = DefaultHasher::new();
    let mut steps = 0;
    while !engine.is_terminal() {
        let legal = engine.legal_actions();
        if legal.is_empty() { break; }
        let a = legal[0];
        engine.step(a);
        a.hash(&mut h);
        steps += 1;
        if steps > 5000 { break; } // safety break, matches Task 22 pattern
    }
    let stats = engine.stats();
    stats.turns_played.hash(&mut h);
    stats.winner_player_id.hash(&mut h);
    h.finish()
}

#[test]
fn regression_hashes_are_stable() {
    // First-run mode: any expected hash of 0 is treated as "fill me in" and
    // the actual hash is printed so it can be copied into this array.
    // After that, any change that flips a hash is a flagged behavioral change.
    let expected = [
        (0u64, 12179243596391117130u64),
        (1u64, 17200018846102743976u64),
        (42u64, 8089009554968948622u64),
        (12345u64, 16772318347194073819u64),
    ];
    for (seed, expected_hash) in expected {
        let actual = game_hash(seed);
        println!("seed {seed} -> {actual}");
        if expected_hash != 0 {
            assert_eq!(actual, expected_hash, "regression for seed {seed}");
        }
    }
}
