//! Phase 0 must not change behavior of legal[0]-policy random self-play.
//! We check that for several seeds, full games complete deterministically and
//! produce a stable event-log hash. Hashes recorded here lock the post-Phase-0
//! seed->trajectory contract going forward.

use catan_engine::Engine;
use std::hash::{Hash, Hasher};

fn play_one(seed: u64) -> u64 {
    let mut e = Engine::new(seed);
    let mut steps = 0;
    while !e.is_terminal() && steps < 5000 {
        if e.is_chance_pending() {
            let outcomes = e.chance_outcomes();
            // Pick the highest-probability outcome deterministically; ties broken by lowest value.
            let (best_val, _) = outcomes.iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap().then_with(|| b.0.cmp(&a.0)))
                .unwrap();
            e.apply_chance_outcome(*best_val);
        } else {
            let legal = e.legal_actions();
            if legal.is_empty() { break; }
            e.step(legal[0]);
        }
        steps += 1;
    }
    let mut h = std::collections::hash_map::DefaultHasher::new();
    for ev in e.events.as_slice() {
        format!("{ev:?}").hash(&mut h);
    }
    h.finish()
}

#[test]
fn fixed_seeds_produce_stable_hashes() {
    let seeds = [1u64, 7, 42, 100, 999];
    // Locked hashes from Step 2 of Task P1.T10. Note: the deterministic policy
    // ("highest-probability chance outcome, ties broken by lowest value" + legal[0])
    // collapses seed-driven variation — board layout is fixed (Board::standard) and
    // both chance points (dice, steal) are overridden by apply_chance_outcome — so all
    // five seeds intentionally produce the same hash. This still locks the post-Phase-0
    // contract: any drift in action encoding, event log shape, legal_actions ordering,
    // or chance-application semantics flips this hash.
    let expected: [u64; 5] = [
        6756322959960289395,
        6756322959960289395,
        6756322959960289395,
        6756322959960289395,
        6756322959960289395,
    ];
    for (s, exp) in seeds.iter().zip(expected.iter()) {
        assert_eq!(play_one(*s), *exp, "seed {s} hash drift");
    }
}
