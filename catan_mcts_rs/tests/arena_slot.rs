//! ArenaSlot must reproduce play_arena_game's game trajectory exactly when
//! driven with the same net outputs: same chance walk (MtRng), same per-seat
//! rngs, same greedy picks. Driving the slot with evaluate_one makes it a
//! B=1 re-encoding of the oracle -> full-game equality is REQUIRED here
//! (same kernels, same RNG streams, only the control flow differs).
use catan_mcts_rs::arena::{play_arena_game, seating_is_cand, ArenaSlot};
use catan_mcts_rs::evaluator::TorchScriptEvaluator;
use catan_mcts_rs::mcts::SessionStep;
use std::path::PathBuf;
use tch::Device;

fn spike() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("mcts_study").join("spike")
}

#[test]
fn slot_driven_b1_equals_oracle() {
    let ts = spike().join("wrapper_traced.ts");
    if !ts.exists() {
        eprintln!("skip: fixture missing");
        return;
    }
    let ev = TorchScriptEvaluator::load(ts.to_str().unwrap(), Device::Cpu).unwrap();
    for (rot, seed) in [(0usize, 42u64), (1, 43), (2, 44), (3, 45)] {
        let seating = seating_is_cand(rot);
        let oracle = play_arena_game(&ev, &ev, seed, seating, 8, 10, true, 5000);
        let mut slot = ArenaSlot::new(rot, seed, 10, true);
        loop {
            match slot.advance_to_search(8, 5000) {
                None => break,
                Some(mut obs) => {
                    loop {
                        let out = ev.evaluate_one(&obs);
                        match slot.provide_cur(&out) {
                            SessionStep::NeedEval(o) => obs = o,
                            SessionStep::Done => break,
                        }
                    }
                    slot.finish_move();
                }
            }
        }
        let got = slot.result.expect("slot finished without result");
        assert_eq!(got.winner_seat, oracle.winner_seat, "rot={rot} seed={seed}");
        assert_eq!(got.timed_out, oracle.timed_out, "rot={rot} seed={seed}");
        assert_eq!(got.vp_margin, oracle.vp_margin, "rot={rot} seed={seed}");
    }
}
