use catan_engine::Engine;

fn run_game(seed: u64) -> (u32, Vec<u32>) {
    let mut engine = Engine::new(seed);
    let mut actions = Vec::new();
    let mut steps = 0;
    while !engine.is_terminal() {
        let legal = engine.legal_actions();
        if legal.is_empty() { break; }
        // Always pick action[0] for full determinism.
        engine.step(legal[0]);
        actions.push(legal[0]);
        steps += 1;
        if steps > 5000 { break; } // Safety break — game may not terminate with greedy-first policy
    }
    let s = engine.stats();
    (s.turns_played, actions)
}

#[test]
fn same_seed_produces_identical_trajectory() {
    let a = run_game(42);
    let b = run_game(42);
    assert_eq!(a, b);
}

#[test]
fn different_seeds_produce_different_trajectories() {
    let a = run_game(1);
    let b = run_game(2);
    assert_ne!(a, b);
}
