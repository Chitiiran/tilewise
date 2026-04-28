//! Top-level smoke test. Plays a complete game using a seeded-random policy.
//! When this passes for 1M random games, v1 is functionally done.
//!
//! Policy: pick a uniformly random action from `legal_actions()` using a
//! dedicated test RNG (seeded so the result is deterministic and reproducible).
//! This is more representative of self-play than always picking legal[0],
//! and ensures EndTurn is sometimes selected even when builds are available.

use catan_engine::Engine;
use rand::rngs::SmallRng;
use rand::{Rng as _, SeedableRng};

#[test]
fn play_random_game_to_completion_without_panicking() {
    let mut engine = Engine::new(42);
    let mut policy_rng = SmallRng::seed_from_u64(0xC47A1B07_u64);
    let mut steps = 0;
    while !engine.is_terminal() {
        if engine.is_chance_pending() {
            // Steal phase has no player legal actions; Roll has only RollDice.
            // Drive both via the chance API to exercise apply_chance_outcome end-to-end.
            let outcomes = engine.chance_outcomes();
            let idx = policy_rng.gen_range(0..outcomes.len());
            engine.apply_chance_outcome(outcomes[idx].0);
        } else {
            let legal = engine.legal_actions();
            assert!(!legal.is_empty(), "no legal actions in non-terminal state");
            let idx = policy_rng.gen_range(0..legal.len());
            let action = legal[idx];
            engine.step(action);
        }
        steps += 1;
        assert!(steps < 5_000, "game did not terminate in 5000 steps");
    }
    assert!(engine.is_terminal());
    println!("Game terminated after {} steps", steps);
}
