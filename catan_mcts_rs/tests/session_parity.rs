//! SearchSession (pausable) must produce IDENTICAL visit_counts + root_value to
//! Mcts::search when driven with the SAME B=1 evaluator and same RNG. This
//! proves the prepare/finish split + state machine is byte-faithful to the
//! recursive search, independent of batching.

use catan_mcts_rs::evaluator::TorchScriptEvaluator;
use catan_mcts_rs::mcts::{Mcts, SearchSession, SessionStep};
use catan_mcts_rs::rng::NpRng;
use catan_engine::Engine;
use std::path::PathBuf;
use tch::Device;

fn spike() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("mcts_study").join("spike")
}

fn drive_session(ev: &TorchScriptEvaluator, engine: Engine, n_sims: u32, seed: u64,
                 eps: f64) -> ([i32; 280], f64) {
    let mut rng = NpRng::from_seed(seed);
    let mut s = SearchSession::new(engine, n_sims, 1.4, 0.8, eps);
    // Contract: first pump() yields the first NeedEval (or Done); thereafter
    // every step is provide(out) -> next NeedEval or Done.
    let mut step = s.pump(&mut rng);
    loop {
        match step {
            SessionStep::NeedEval(obs) => {
                let out = ev.evaluate_one(&obs);
                step = s.provide(&out, &mut rng);
            }
            SessionStep::Done => break,
        }
    }
    (s.take_visit_counts(), s.last_root_value)
}

fn advance_to_decision(seed: u64, plies: u32) -> Engine {
    let mut e = Engine::new(seed);
    let mut rng = NpRng::from_seed(seed ^ 0xABCD);
    let mut taken = 0;
    while !e.is_terminal() && taken < plies {
        if e.is_chance_pending() {
            let v = catan_mcts_rs::state::sample_chance(&e, &mut rng);
            e.apply_chance_outcome(v);
            continue;
        }
        let legal = e.legal_actions();
        if legal.len() == 1 { e.step(legal[0]); continue; }
        let i = (rng.random_f64() * legal.len() as f64) as usize;
        e.step(legal[i.min(legal.len() - 1)]);
        taken += 1;
    }
    e
}

#[test]
fn session_matches_recursive_search() {
    let single = spike().join("wrapper_traced.ts");
    if !single.exists() { eprintln!("skip: {single:?} missing"); return; }
    let ev = TorchScriptEvaluator::load(single.to_str().unwrap(), Device::Cpu).unwrap();

    for (seed, plies, n_sims, eps) in [
        (101u64, 6u32, 8u32, 0.0f64),
        (202, 10, 16, 0.0),
        (303, 8, 12, 0.25),
        (404, 5, 16, 0.25),
    ] {
        let engine = advance_to_decision(seed, plies);
        if engine.is_terminal() { continue; }

        // Recursive search.
        let mut rng_a = NpRng::from_seed(seed + 7);
        let mut mcts = Mcts::new(&ev, 1.4, 0.8, eps);
        let vc_search = mcts.search(engine.clone(), n_sims, &mut rng_a);
        let rv_search = mcts.last_root_value;

        // Session, SAME rng seed.
        let (vc_sess, rv_sess) = drive_session(&ev, engine, n_sims, seed + 7, eps);

        assert_eq!(vc_sess.to_vec(), vc_search.to_vec(),
                   "visit_counts differ (seed={seed}, sims={n_sims}, eps={eps})");
        assert_eq!(rv_sess.to_bits(), rv_search.to_bits(), "root_value differs");
    }
}
