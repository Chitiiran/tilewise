//! Top-level smoke test. Fails until the engine can play a complete game.
//! When this passes for 1M random games, v1 is functionally done.

use catan_engine::Engine;

#[test]
fn play_random_game_to_completion_without_panicking() {
    let mut engine = Engine::new(42);
    let mut steps = 0;
    while !engine.is_terminal() {
        let legal = engine.legal_actions();
        assert!(!legal.is_empty(), "no legal actions in non-terminal state");
        let action = legal[0]; // deterministic choice for now
        engine.step(action);
        steps += 1;
        assert!(steps < 5_000, "game did not terminate in 5000 steps");
    }
    assert!(engine.is_terminal());
}
