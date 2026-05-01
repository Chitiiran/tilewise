use catan_engine::Engine;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

fn game_hash(seed: u64) -> u64 {
    let mut engine = Engine::with_standard_board(seed);
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
    // v2 hashes (post-trade-cap): the per-turn trade cap (MAX_TRADES_PER_TURN=4)
    // broke the pathological convergence to a single attractor that earlier v2
    // had — different seeds now produce different trajectories because the
    // legal[0] policy can no longer get stuck in an infinite ProposeTrade loop.
    // Post-P2 (real port positions). The new perimeter port layout shifted
    // setup-phase placement value, which the deterministic legal[0] policy
    // reacts to — different seeds now produce 3 distinct hashes (better
    // diversity than the pre-trade-cap single-attractor regime).
    // Post-fix-138 (terminal check + trade cap=1).
    let expected = [
        (0u64,     13510722627221345023u64),
        (1u64,     4550780657234542719u64),
        (42u64,    16159416546797469512u64),
        (12345u64, 16159416546797469512u64),
    ];
    for (seed, expected_hash) in expected {
        let actual = game_hash(seed);
        println!("seed {seed} -> {actual}");
        if expected_hash != 0 {
            assert_eq!(actual, expected_hash, "regression for seed {seed}");
        }
    }
}
